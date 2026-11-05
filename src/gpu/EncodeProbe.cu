#include "EncodeProbe.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/cub.cuh>

namespace
{

void cudaCheck(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
}

using fastibs::HashParams;
using fastibs::ProbeParams;
using fastibs::WindowDesc;

// Invalid/padding marker in the flat code arrays. A canonical code can
// never be all ones (see HashCore.hpp), and its low 2k bits (the radix
// sort's key range) are the maximum of that range, so sentinels sort last.
constexpr uint64_t SORT_SENTINEL = ~0ull;
constexpr int MAX_SHARDS = 16;
constexpr int BLOCK = 256;
constexpr int PART_ITEMS = 8; // positions per thread per partition tile

// POD probe functors, so kernels can be templated over the backend. Both
// carry their shard's range and answer other shards' queries with false
// (unreachable through the routed pipeline; a guard for other callers).
template <typename RecT>
struct BsProbe
{
    const uint64_t *lut;
    const RecT *suffixes;
    const uint32_t *sigMap;
    const uint32_t *norm;
    ProbeParams P;
    __device__ bool operator()(uint64_t code) const
    {
        return fastibs::probeOne<RecT>(code, lut, suffixes, sigMap, norm, P);
    }
};

struct HashProbe
{
    const uint64_t *table;
    HashParams P;
    __device__ bool operator()(uint64_t code) const
    {
        return fastibs::hashContains(code, table, P);
    }
};

template <typename Fn>
void withProbe(bool useHash, const DeviceHashIndex::DeviceView &hv,
               const DeviceKmerIndex::DeviceView &bv, Fn &&fn)
{
    if (useHash)
        fn(HashProbe{hv.table, hv.params});
    else if (bv.rec32)
        fn(BsProbe<uint32_t>{bv.lut, (const uint32_t *)bv.suffix, bv.sigMap,
                             bv.norm, bv.params});
    else
        fn(BsProbe<uint64_t>{bv.lut, (const uint64_t *)bv.suffix, bv.sigMap,
                             bv.norm, bv.params});
}

// Panel mode: the n databases' shards resident on one device, probed from
// ONE encode (fastibs::encodeProbeWordMulti). prepare() computes the
// minimizer signature once per code when the databases share k and the
// signature length (their norm tables are then identical, so database 0's
// serves all); otherwise every database computes its own inside probeOne.
// The record width is a per-database property (32-bit suffix records when
// (k - p)/4 <= 4 bytes), a warp-uniform branch.
struct MultiBsProbe
{
    DeviceKmerIndex::DeviceView v[fastibs::MAX_DBS];
    uint32_t n;
    uint32_t k;
    uint32_t sigLen; // the shared signature length (0 for KMC1)
    bool sharedSig;
    const uint32_t *norm; // database 0's norm table
    __device__ uint32_t count() const { return n; }
    __device__ uint32_t prepare(uint64_t code) const
    {
        return (sharedSig && sigLen) ? fastibs::signatureOf(code, k, sigLen, norm) : 0;
    }
    __device__ bool operator()(uint32_t d, uint64_t code, uint32_t sig) const
    {
        const DeviceKmerIndex::DeviceView &x = v[d];
        if (sharedSig)
            return x.rec32 ? fastibs::probeOneWithSig<uint32_t>(
                                 code, sig, x.lut, (const uint32_t *)x.suffix, x.sigMap, x.params)
                           : fastibs::probeOneWithSig<uint64_t>(
                                 code, sig, x.lut, (const uint64_t *)x.suffix, x.sigMap, x.params);
        return x.rec32 ? fastibs::probeOne<uint32_t>(code, x.lut, (const uint32_t *)x.suffix,
                                                     x.sigMap, x.norm, x.params)
                       : fastibs::probeOne<uint64_t>(code, x.lut, (const uint64_t *)x.suffix,
                                                     x.sigMap, x.norm, x.params);
    }
};

// Owner functors: which shard holds a canonical code. The ported index is
// split by prefix range (the top 2p bits of the code, a handful of
// compares); the hash backend by its owner hash.
struct OwnerByPrefix
{
    uint32_t shift;
    uint32_t n;
    uint32_t cutHi[MAX_SHARDS]; // shard d holds prefixes [cutHi[d-1], cutHi[d])
    __device__ int operator()(uint64_t code) const
    {
        const uint32_t prefix = (uint32_t)(code >> shift);
        int d = 0;
        while (d + 1 < (int)n && prefix >= cutHi[d])
            ++d;
        return d;
    }
};

struct OwnerByHash
{
    uint32_t n;
    __device__ int operator()(uint64_t code) const
    {
        return (int)fastibs::hashShardOf(code, n);
    }
};

template <typename Fn>
void withOwner(bool useHash, const OwnerByPrefix &byPrefix,
               const OwnerByHash &byHash, Fn &&fn)
{
    if (useHash)
        fn(byHash);
    else
        fn(byPrefix);
}

// ---------------------------------------------------------------------------
// Single-device kernels (unchanged from Phase 2/3).

// Fused K1+K2: one thread per 32-position output word, grid-stride. The
// body is the shared host/device template from EncodeProbeCore.hpp.
template <typename ProbeFn>
__global__ void encodeProbeKernel(const WindowDesc *__restrict__ descs,
                                  uint32_t nWindows,
                                  const uint64_t *__restrict__ seqWords,
                                  const uint64_t *__restrict__ invWords,
                                  uint64_t nOutWords, uint32_t k,
                                  ProbeFn probe,
                                  uint32_t *__restrict__ hitWords,
                                  uint32_t *__restrict__ validWords)
{
    for (uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         w < nOutWords; w += (uint64_t)gridDim.x * blockDim.x)
    {
        uint32_t h, v;
        fastibs::encodeProbeWord(w, descs, nWindows, seqWords, invWords, k,
                                 probe, h, v);
        hitWords[w] = h;
        validWords[w] = v;
    }
}

// Panel mode, one device: K1 once, K2 for each of the n databases, hit
// masks database-major with stride hitStride.
template <typename MultiProbeFn>
__global__ void encodeProbeMultiKernel(const WindowDesc *__restrict__ descs,
                                       uint32_t nWindows,
                                       const uint64_t *__restrict__ seqWords,
                                       const uint64_t *__restrict__ invWords,
                                       uint64_t nOutWords, uint32_t k,
                                       MultiProbeFn probe,
                                       uint32_t *__restrict__ hitWords, size_t hitStride,
                                       uint32_t *__restrict__ validWords)
{
    for (uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         w < nOutWords; w += (uint64_t)gridDim.x * blockDim.x)
    {
        uint32_t v;
        fastibs::encodeProbeWordMulti(w, descs, nWindows, seqWords, invWords, k,
                                      probe, hitWords, hitStride, v);
        validWords[w] = v;
    }
}

// Sorted path, stage 1 (K1 only): emit each position's canonical code into
// a flat array indexed by output-bit position (invalid positions emit the
// sentinel; padding entries were pre-set to the sentinel by a memset), and
// write the validity words.
__global__ void encodeCollectKernel(const WindowDesc *__restrict__ descs,
                                    uint32_t nWindows,
                                    const uint64_t *__restrict__ seqWords,
                                    const uint64_t *__restrict__ invWords,
                                    uint64_t nOutWords, uint32_t k,
                                    uint64_t *__restrict__ codes,
                                    uint32_t *__restrict__ validWords)
{
    for (uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         w < nOutWords; w += (uint64_t)gridDim.x * blockDim.x)
    {
        const uint64_t base = w * 32;
        validWords[w] = fastibs::encodeWord(
            w, descs, nWindows, seqWords, invWords, k,
            [&](uint32_t bit, uint64_t code, bool ok)
            { codes[base + bit] = ok ? code : SORT_SENTINEL; });
    }
}

__global__ void iotaKernel(uint32_t *__restrict__ pos, size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        pos[i] = (uint32_t)i;
}

// Sorted path, stage 3 (K2): probe the value-sorted keys and scatter hits
// back through the position payload.
template <typename ProbeFn>
__global__ void probeScatterKernel(const uint64_t *__restrict__ codes,
                                   const uint32_t *__restrict__ pos, size_t n,
                                   ProbeFn probe,
                                   uint32_t *__restrict__ hitWords)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
    {
        const uint64_t code = __ldg(codes + i);
        if (code == SORT_SENTINEL)
            continue;
        if (probe(code))
        {
            const uint32_t p = __ldg(pos + i);
            atomicOr((unsigned int *)&hitWords[p >> 5], 1u << (p & 31));
        }
    }
}

// ---------------------------------------------------------------------------
// Routed kernels (several devices). All run on the encoder except
// probeSegmentKernel, which runs on each owner — on the encoder's memory
// when peer access exists, on a staged copy otherwise.

// K1 + owner count: one thread per output word emits the flat code array
// and the validity words, tallies its codes' owners in registers, and the
// block flushes one atomicAdd per owner.
template <typename OwnerFn>
__global__ void routeKernel(const WindowDesc *__restrict__ descs,
                            uint32_t nWindows,
                            const uint64_t *__restrict__ seqWords,
                            const uint64_t *__restrict__ invWords,
                            uint64_t nOutWords, uint32_t k, OwnerFn owner,
                            uint64_t *__restrict__ codes,
                            uint32_t *__restrict__ validWords,
                            uint32_t *__restrict__ counts)
{
    __shared__ uint32_t s_cnt[MAX_SHARDS];
    if (threadIdx.x < MAX_SHARDS)
        s_cnt[threadIdx.x] = 0;
    __syncthreads();

    uint32_t local[MAX_SHARDS];
    for (int o = 0; o < MAX_SHARDS; ++o)
        local[o] = 0;
    for (uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         w < nOutWords; w += (uint64_t)gridDim.x * blockDim.x)
    {
        const uint64_t base = w * 32;
        validWords[w] = fastibs::encodeWord(
            w, descs, nWindows, seqWords, invWords, k,
            [&](uint32_t bit, uint64_t code, bool ok)
            {
                codes[base + bit] = ok ? code : SORT_SENTINEL;
                if (ok)
                    ++local[owner(code)];
            });
    }
    for (uint32_t o = 0; o < owner.n; ++o)
        if (local[o])
            atomicAdd(&s_cnt[o], local[o]);
    __syncthreads();
    if (threadIdx.x < owner.n)
        atomicAdd(&counts[threadIdx.x], s_cnt[threadIdx.x]);
}

// Exclusive scan of the owner counts: segment d of the partition is
// [segOff[d], segOff[d+1]); the cursors start at the segment starts.
__global__ void segmentOffsetsKernel(const uint32_t *__restrict__ counts,
                                     uint32_t n, uint32_t *__restrict__ segOff,
                                     uint32_t *__restrict__ cursor)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        uint32_t acc = 0;
        for (uint32_t o = 0; o < n; ++o)
        {
            segOff[o] = acc;
            cursor[o] = acc;
            acc += counts[o];
        }
        segOff[n] = acc;
    }
}

// Partition: each block takes a tile of positions, counts its owners in
// shared memory, reserves one range per owner with a single atomicAdd on
// that owner's cursor, and scatters (code, position) into the reserved
// ranges. Order within a segment is arbitrary; hits are ORed by position.
template <typename OwnerFn>
__global__ void partitionKernel(const uint64_t *__restrict__ codes,
                                uint64_t nPositions, OwnerFn owner,
                                uint32_t *__restrict__ cursor,
                                uint64_t *__restrict__ partCodes,
                                uint32_t *__restrict__ partPos)
{
    __shared__ uint32_t s_cnt[MAX_SHARDS];
    __shared__ uint32_t s_base[MAX_SHARDS];
    const uint64_t tile = (uint64_t)blockDim.x * PART_ITEMS;
    for (uint64_t t0 = (uint64_t)blockIdx.x * tile; t0 < nPositions;
         t0 += (uint64_t)gridDim.x * tile)
    {
        if (threadIdx.x < MAX_SHARDS)
            s_cnt[threadIdx.x] = 0;
        __syncthreads();
        uint64_t myCode[PART_ITEMS];
        int myOwner[PART_ITEMS];
        for (int j = 0; j < PART_ITEMS; ++j)
        {
            const uint64_t i = t0 + (uint64_t)j * blockDim.x + threadIdx.x;
            uint64_t c = SORT_SENTINEL;
            if (i < nPositions)
                c = codes[i];
            myCode[j] = c;
            myOwner[j] = -1;
            if (c != SORT_SENTINEL)
            {
                myOwner[j] = owner(c);
                atomicAdd(&s_cnt[myOwner[j]], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x < owner.n)
        {
            s_base[threadIdx.x] = atomicAdd(&cursor[threadIdx.x], s_cnt[threadIdx.x]);
            s_cnt[threadIdx.x] = 0; // reused as the in-tile cursor
        }
        __syncthreads();
        for (int j = 0; j < PART_ITEMS; ++j)
        {
            if (myOwner[j] < 0)
                continue;
            const uint32_t idx = s_base[myOwner[j]] + atomicAdd(&s_cnt[myOwner[j]], 1u);
            partCodes[idx] = myCode[j];
            partPos[idx] = (uint32_t)(t0 + (uint64_t)j * blockDim.x + threadIdx.x);
        }
        __syncthreads();
    }
}

// Owner-side K2 over one segment of the encoder's partition: one hit flag
// per query, written back into the encoder's flag lane. No __ldg: the
// pointers may be peer memory.
template <typename ProbeFn>
__global__ void probeSegmentKernel(const uint64_t *codes, const uint32_t *segOff,
                                   int owner, ProbeFn probe, uint8_t *flags)
{
    const uint32_t lo = segOff[owner], hi = segOff[owner + 1];
    for (uint32_t i = lo + blockIdx.x * blockDim.x + threadIdx.x; i < hi;
         i += gridDim.x * blockDim.x)
        flags[i] = probe(codes[i]) ? 1 : 0;
}

// Panel mode: the owner probes its segment against its shard of every
// database, n flags per query (lane d of this owner's block at
// flags + d * dbStride), with the signature computed once per query.
template <typename MultiProbeFn>
__global__ void probeSegmentMultiKernel(const uint64_t *codes, const uint32_t *segOff,
                                        int owner, MultiProbeFn probe, uint8_t *flags,
                                        size_t dbStride)
{
    const uint32_t lo = segOff[owner], hi = segOff[owner + 1];
    const uint32_t n = probe.count();
    for (uint32_t i = lo + blockIdx.x * blockDim.x + threadIdx.x; i < hi;
         i += gridDim.x * blockDim.x)
    {
        const uint64_t code = codes[i];
        const uint32_t prep = probe.prepare(code);
        for (uint32_t d = 0; d < n; ++d)
            flags[(size_t)d * dbStride + i] = probe(d, code, prep) ? 1 : 0;
    }
}

// Panel mode: flags -> n position-indexed hit masks. ownerStride is 0 when
// every owner wrote into the one shared block (peer access), the block
// size when each owner has its own (staged copies); within a block lane d
// is at d * dbStride.
__global__ void scatterHitsMultiKernel(const uint8_t *__restrict__ flags,
                                       size_t ownerStride, size_t dbStride, uint32_t nDbs,
                                       const uint32_t *__restrict__ partPos,
                                       const uint32_t *__restrict__ segOff, int n,
                                       uint32_t *__restrict__ hitWords, size_t hitStride)
{
    const uint32_t total = segOff[n];
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x)
    {
        int o = 0;
        while (o + 1 < n && i >= segOff[o + 1])
            ++o;
        const uint32_t p = partPos[i];
        const uint8_t *lane = flags + (size_t)o * ownerStride + i;
        for (uint32_t d = 0; d < nDbs; ++d)
            if (lane[(size_t)d * dbStride])
                atomicOr((unsigned int *)&hitWords[(size_t)d * hitStride + (p >> 5)],
                         1u << (p & 31));
    }
}

// Encoder-side assembly: flags -> position-indexed hit mask. laneStride
// is 0 when every owner wrote into the one shared flag array (peer
// access), maxPositions when each owner has its own lane (staged copies).
__global__ void scatterHitsKernel(const uint8_t *__restrict__ flags,
                                  size_t laneStride,
                                  const uint32_t *__restrict__ partPos,
                                  const uint32_t *__restrict__ segOff, int n,
                                  uint32_t *__restrict__ hitWords)
{
    const uint32_t total = segOff[n];
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x)
    {
        int o = 0;
        while (o + 1 < n && i >= segOff[o + 1])
            ++o;
        if (flags[(size_t)o * laneStride + i])
        {
            const uint32_t p = partPos[i];
            atomicOr((unsigned int *)&hitWords[p >> 5], 1u << (p & 31));
        }
    }
}

int gridFor(size_t work, int perBlock)
{
    return (int)std::max<size_t>(
        1, std::min<size_t>((work + perBlock - 1) / perBlock, 65535));
}

template <typename T>
T *deviceAlloc(size_t bytes, const char *what, size_t &tally)
{
    T *p = nullptr;
    cudaCheck(cudaMalloc(&p, bytes), what);
    tally += bytes;
    return p;
}

} // namespace

// One device's part of one slot.
struct EncodeProbeEngine::DevSlot
{
    uint64_t *d_seq = nullptr;
    uint64_t *d_inv = nullptr;
    WindowDesc *d_desc = nullptr;
    uint32_t *d_hit = nullptr;
    uint32_t *d_valid = nullptr;
    // encoder role (several devices): flat codes, the per-shard partition,
    // the owners' hit flags, the counters
    uint64_t *d_codes = nullptr;
    uint64_t *d_partCodes = nullptr;
    uint32_t *d_partPos = nullptr;
    uint8_t *d_flags = nullptr;
    uint32_t *d_counts = nullptr;
    uint32_t *d_segOff = nullptr;
    uint32_t *d_cursor = nullptr;
    // owner role without peer access: staged copy of the encoder's partition
    uint64_t *d_recvCodes = nullptr;
    uint32_t *d_recvSegOff = nullptr;
    uint8_t *d_localFlags = nullptr;
    // single-device sorted path
    uint64_t *d_codesAlt = nullptr;
    uint32_t *d_pos = nullptr;
    uint32_t *d_posAlt = nullptr;
    void *d_cubTemp = nullptr;

    cudaStream_t encStream = nullptr; // this device as encoder (or alone)
    cudaStream_t ownStream = nullptr; // this device as owner for another encoder
    cudaEvent_t evRouted = nullptr;   // encoder: partition ready
    cudaEvent_t evProbed = nullptr;   // owner: flags written
    cudaEvent_t evDone = nullptr;     // encoder: masks in the host buffers
};

struct EncodeProbeEngine::Dev
{
    int device = 0;
    DeviceKmerIndex::DeviceView bs{}; // database 0's shard on this device
    DeviceHashIndex::DeviceView hash{};
    MultiBsProbe multi{}; // panel mode: every database's shard on this device
    size_t cubTempBytes = 0;
    std::vector<DevSlot> slots;
};

struct EncodeProbeEngine::Slot
{
    // pinned host staging, portable: any device may copy in or out
    uint64_t *h_seq = nullptr;
    uint64_t *h_inv = nullptr;
    WindowDesc *h_desc = nullptr;
    uint32_t *h_hit = nullptr;
    uint32_t *h_valid = nullptr;
    int encoder = 0; // device of the in-flight batch
    bool inFlight = false;
};

EncodeProbeEngine::EncodeProbeEngine(const std::vector<const DeviceKmerIndex *> &bs,
                                     const std::vector<const DeviceHashIndex *> &hash,
                                     uint32_t kmerSize, size_t maxBatchBases,
                                     size_t maxBatchWindows, int depth,
                                     bool sortQueries, bool pinLut)
    : nSlots(depth), k(kmerSize), sorted(sortQueries), useHash(!hash.empty())
{
    if (bs.empty() == hash.empty())
        throw std::runtime_error(
            "EncodeProbeEngine: exactly one probe backend must be given");
    cap.windows = maxBatchWindows;
    cap.seqWords = maxBatchBases / 32 + maxBatchWindows + 1;
    cap.invWords = maxBatchBases / 64 + maxBatchWindows + 1;
    cap.outWords = maxBatchBases / 32 + maxBatchWindows + 1;
    std::vector<std::vector<const DeviceKmerIndex *>> perDb;
    if (!bs.empty())
        perDb.push_back(bs);
    init(perDb, hash, pinLut);
}

EncodeProbeEngine::EncodeProbeEngine(const std::vector<std::vector<const DeviceKmerIndex *>> &bs,
                                     uint32_t kmerSize, size_t maxBatchBases,
                                     size_t maxBatchWindows, int depth, bool pinLut)
    : nSlots(depth), k(kmerSize), sorted(false), useHash(false)
{
    if (bs.empty() || bs[0].empty())
        throw std::runtime_error("EncodeProbeEngine: no database given");
    cap.windows = maxBatchWindows;
    cap.seqWords = maxBatchBases / 32 + maxBatchWindows + 1;
    cap.invWords = maxBatchBases / 64 + maxBatchWindows + 1;
    cap.outWords = maxBatchBases / 32 + maxBatchWindows + 1;
    init(bs, {}, pinLut);
}

// bs[i][d]: shard d of database i (empty for the hash backend). Capacities
// (cap) are set by the constructors; this does everything else.
void EncodeProbeEngine::init(const std::vector<std::vector<const DeviceKmerIndex *>> &bs,
                             const std::vector<const DeviceHashIndex *> &hash, bool pinLut)
{
    nDbs = useHash ? 1 : (int)bs.size();
    if (nDbs > (int)fastibs::MAX_DBS)
        throw std::runtime_error("EncodeProbeEngine: at most " +
                                 std::to_string(fastibs::MAX_DBS) + " databases at once");
    nDevs = (int)(useHash ? hash.size() : bs[0].size());
    if (nDevs > MAX_SHARDS)
        throw std::runtime_error("EncodeProbeEngine: at most 16 devices");
    for (const auto &shards : bs)
        if ((int)shards.size() != nDevs)
            throw std::runtime_error("EncodeProbeEngine: every database needs one shard per device");

    // Each window pads its sequence/exception/output regions to a word
    // boundary, so capacities carry one extra word per window.
    maxPositions = cap.outWords * 32;
    if ((sorted || nDevs > 1) && maxPositions > (size_t)UINT32_MAX)
        throw std::runtime_error(
            "EncodeProbeEngine: batch too large (" + std::to_string(maxPositions) +
            " positions exceed the uint32 position payload); reduce --batch-mb");
    if (sorted && nDevs > 1)
    {
        fprintf(stderr, "[gpu] --sort-queries is a single-device option; ignored with %d devices\n", nDevs);
        sorted = false;
    }

    devs = new Dev[nDevs];
    // Owner function of the ported index. With several databases every one
    // is split with the same cut (KmcRaIndex::jointShardPlan) expressed at
    // the coarsest prefix length among them, so one function serves all.
    uint32_t pMin = 0;
    if (!useHash)
    {
        pMin = bs[0][0]->view().params.p;
        for (const auto &shards : bs)
        {
            if (shards[0]->view().params.k != k)
                throw std::runtime_error("EncodeProbeEngine: the databases do not share k");
            pMin = std::min(pMin, shards[0]->view().params.p);
        }
    }
    for (int d = 0; d < nDevs; ++d)
    {
        if (useHash)
        {
            devs[d].device = hash[d]->device();
            devs[d].hash = hash[d]->view();
        }
        else
        {
            devs[d].device = bs[0][d]->device();
            devs[d].bs = bs[0][d]->view();
            const uint32_t p0 = bs[0][d]->view().params.p;
            ownerCutHi[d] = bs[0][d]->shard().prefixHi >> (2 * (p0 - pMin));
            MultiBsProbe &m = devs[d].multi;
            m.n = (uint32_t)nDbs;
            m.k = k;
            m.sigLen = bs[0][d]->view().params.sigLen;
            m.sharedSig = true;
            m.norm = bs[0][d]->view().norm;
            for (int i = 0; i < nDbs; ++i)
            {
                const DeviceKmerIndex *s = bs[i][d];
                if (s->device() != devs[d].device)
                    throw std::runtime_error("EncodeProbeEngine: database shards of one device index differ in device");
                const uint32_t pi = s->view().params.p;
                if ((s->shard().prefixHi >> (2 * (pi - pMin))) != ownerCutHi[d] ||
                    (s->shard().prefixLo >> (2 * (pi - pMin))) !=
                        (d ? ownerCutHi[d - 1] : 0u))
                    throw std::runtime_error("EncodeProbeEngine: the databases are not split with one cut");
                m.v[i] = s->view();
                if (s->view().params.sigLen != m.sigLen)
                    m.sharedSig = false;
            }
        }
        devs[d].slots.resize(nSlots);
    }
    if (!useHash)
        ownerShift = 2 * (k - pMin);

    slots = new Slot[nSlots];
    try
    {
        direct = nDevs > 1 ? enablePeerAccess() : false;
        if (nDevs > 1)
            fprintf(stderr, "[gpu] routing: %s\n", direct ? "peer" : "staged");

        cudaCheck(cudaSetDevice(devs[0].device), "cudaSetDevice");
        for (int s = 0; s < nSlots; ++s)
        {
            Slot &sl = slots[s];
            cudaCheck(cudaHostAlloc(&sl.h_seq, cap.seqWords * 8, cudaHostAllocPortable), "pinned seq alloc");
            cudaCheck(cudaHostAlloc(&sl.h_inv, cap.invWords * 8, cudaHostAllocPortable), "pinned inv alloc");
            cudaCheck(cudaHostAlloc(&sl.h_desc, cap.windows * sizeof(WindowDesc), cudaHostAllocPortable), "pinned desc alloc");
            cudaCheck(cudaHostAlloc(&sl.h_hit, (size_t)nDbs * cap.outWords * 4, cudaHostAllocPortable), "pinned hit alloc");
            cudaCheck(cudaHostAlloc(&sl.h_valid, cap.outWords * 4, cudaHostAllocPortable), "pinned valid alloc");
        }

        for (int d = 0; d < nDevs; ++d)
        {
            Dev &dv = devs[d];
            cudaCheck(cudaSetDevice(dv.device), "cudaSetDevice");
            size_t bytes = 0;
            if (sorted && dv.cubTempBytes == 0)
            {
                cub::DoubleBuffer<uint64_t> keys(nullptr, nullptr);
                cub::DoubleBuffer<uint32_t> vals(nullptr, nullptr);
                cudaCheck(cub::DeviceRadixSort::SortPairs(nullptr, dv.cubTempBytes, keys, vals,
                                                          maxPositions, 0, 2 * (int)k),
                          "cub temp query");
            }
            for (int s = 0; s < nSlots; ++s)
            {
                DevSlot &ds = dv.slots[s];
                ds.d_seq = deviceAlloc<uint64_t>(cap.seqWords * 8, "device seq alloc", bytes);
                ds.d_inv = deviceAlloc<uint64_t>(cap.invWords * 8, "device inv alloc", bytes);
                ds.d_desc = deviceAlloc<WindowDesc>(cap.windows * sizeof(WindowDesc), "device desc alloc", bytes);
                ds.d_hit = deviceAlloc<uint32_t>((size_t)nDbs * cap.outWords * 4, "device hit alloc", bytes);
                ds.d_valid = deviceAlloc<uint32_t>(cap.outWords * 4, "device valid alloc", bytes);
                if (nDevs > 1)
                {
                    ds.d_codes = deviceAlloc<uint64_t>(maxPositions * 8, "codes alloc", bytes);
                    ds.d_partCodes = deviceAlloc<uint64_t>(maxPositions * 8, "partition codes alloc", bytes);
                    ds.d_partPos = deviceAlloc<uint32_t>(maxPositions * 4, "partition pos alloc", bytes);
                    ds.d_flags = deviceAlloc<uint8_t>(maxPositions * (size_t)nDbs * (direct ? 1 : nDevs), "flags alloc", bytes);
                    ds.d_counts = deviceAlloc<uint32_t>(MAX_SHARDS * 4, "counts alloc", bytes);
                    ds.d_segOff = deviceAlloc<uint32_t>((MAX_SHARDS + 1) * 4, "segment offsets alloc", bytes);
                    ds.d_cursor = deviceAlloc<uint32_t>(MAX_SHARDS * 4, "cursor alloc", bytes);
                    if (!direct)
                    {
                        ds.d_recvCodes = deviceAlloc<uint64_t>(maxPositions * 8, "staged codes alloc", bytes);
                        ds.d_recvSegOff = deviceAlloc<uint32_t>((MAX_SHARDS + 1) * 4, "staged offsets alloc", bytes);
                        ds.d_localFlags = deviceAlloc<uint8_t>(maxPositions * (size_t)nDbs, "staged flags alloc", bytes);
                    }
                    cudaCheck(cudaStreamCreateWithFlags(&ds.ownStream, cudaStreamNonBlocking), "owner stream create");
                    cudaCheck(cudaEventCreateWithFlags(&ds.evRouted, cudaEventDisableTiming), "event create");
                    cudaCheck(cudaEventCreateWithFlags(&ds.evProbed, cudaEventDisableTiming), "event create");
                }
                else if (sorted)
                {
                    ds.d_codes = deviceAlloc<uint64_t>(maxPositions * 8, "device codes alloc", bytes);
                    ds.d_codesAlt = deviceAlloc<uint64_t>(maxPositions * 8, "device codes-alt alloc", bytes);
                    ds.d_pos = deviceAlloc<uint32_t>(maxPositions * 4, "device pos alloc", bytes);
                    ds.d_posAlt = deviceAlloc<uint32_t>(maxPositions * 4, "device pos-alt alloc", bytes);
                    ds.d_cubTemp = deviceAlloc<void>(dv.cubTempBytes, "cub temp alloc", bytes);
                }
                cudaCheck(cudaStreamCreateWithFlags(&ds.encStream, cudaStreamNonBlocking), "stream create");
                cudaCheck(cudaEventCreateWithFlags(&ds.evDone, cudaEventDisableTiming), "event create");
                if (pinLut && !useHash)
                {
                    // One access-policy window per stream: database 0's LUT.
                    bs[0][d]->setLutPersisting(ds.encStream);
                    if (ds.ownStream)
                        bs[0][d]->setLutPersisting(ds.ownStream);
                }
            }
            perDeviceBytes = std::max(perDeviceBytes, bytes);
        }
    }
    catch (...)
    {
        release();
        throw;
    }
}

// The constructor's allocations, as a number: one slot's buffers times
// the depth. Kept next to the constructor so the two cannot drift.
size_t EncodeProbeEngine::estimateDeviceBytes(int nDevs, bool peerDirect,
                                              size_t maxBatchBases,
                                              size_t maxBatchWindows, int depth,
                                              bool sortQueries, int nDbs)
{
    const size_t seqWords = maxBatchBases / 32 + maxBatchWindows + 1;
    const size_t invWords = maxBatchBases / 64 + maxBatchWindows + 1;
    const size_t outWords = maxBatchBases / 32 + maxBatchWindows + 1;
    const size_t P = outWords * 32;
    const size_t D = (size_t)std::max(1, nDbs);
    size_t slot = seqWords * 8 + invWords * 8 + maxBatchWindows * sizeof(WindowDesc) +
                  outWords * 4 * (1 + D);
    if (nDevs > 1)
    {
        // codes, partition codes + positions, flags (one lane per database,
        // one block of lanes per owner without peer access)
        slot += P * (8 + 8 + 4) + P * D * (peerDirect ? 1 : (size_t)nDevs) +
                (MAX_SHARDS * 3 + 1) * 4;
        if (!peerDirect) // staged copy of the partition + local flags
            slot += P * 8 + (MAX_SHARDS + 1) * 4 + P * D;
    }
    else if (sortQueries) // codes, codes-alt, pos, pos-alt (+ a small CUB scratch)
        slot += P * (8 + 8 + 4 + 4);
    return slot * (size_t)std::max(1, depth);
}

namespace
{
bool stagedForcedFlag = false; // --no-peer
}

void EncodeProbeEngine::forceStaged()
{
    stagedForcedFlag = true;
}

bool EncodeProbeEngine::stagedForced()
{
    static const bool env = []
    {
        const char *v = getenv("FASTIBS_NO_PEER");
        return v && *v && std::string(v) != "0";
    }();
    return stagedForcedFlag || env;
}

bool EncodeProbeEngine::peerAccessible(const std::vector<int> &devices)
{
    if (stagedForced())
        return false;
    for (size_t a = 0; a < devices.size(); ++a)
        for (size_t b = 0; b < devices.size(); ++b)
        {
            if (a == b)
                continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, devices[a], devices[b]) != cudaSuccess)
            {
                cudaGetLastError();
                return false;
            }
            if (!can)
                return false;
        }
    return true;
}

// Peer access between every pair of devices lets an owner's kernel read
// the encoder's partition and write its flags in place. Logs the topology;
// which pairs are NVLink is visible in `nvidia-smi topo -m` (the CUDA
// performance rank orders the links: lower is faster).
bool EncodeProbeEngine::enablePeerAccess()
{
    if (stagedForced())
    {
        fprintf(stderr,
                "[gpu] peer access disabled (--no-peer / FASTIBS_NO_PEER=1): queries and hit "
                "flags are staged through host memory\n");
        return false;
    }
    bool all = true;
    for (int a = 0; a < nDevs; ++a)
        for (int b = 0; b < nDevs; ++b)
        {
            if (a == b)
                continue;
            int can = 0;
            cudaCheck(cudaDeviceCanAccessPeer(&can, devs[a].device, devs[b].device),
                      "cudaDeviceCanAccessPeer");
            int rank = -1;
            cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank,
                                      devs[a].device, devs[b].device);
            cudaGetLastError();
            fprintf(stderr, "[gpu] peer %d -> %d: %s (link rank %d)\n", devs[a].device,
                    devs[b].device, can ? "direct access" : "no peer access", rank);
            all = all && can;
        }
    if (!all)
    {
        fprintf(stderr,
                "[gpu] no peer access on some pair: queries and hit flags are staged "
                "through host memory\n");
        return false;
    }
    // Enabling can still fail where the query said yes: a device runs out of
    // peer resources (cudaErrorTooManyPeers), or the driver refuses a
    // pair. The staged path needs no peer mapping (cudaMemcpyPeerAsync
    // goes through the driver either way), so any failure falls back to it
    // instead of aborting the reference; pairs already enabled are
    // harmless to it.
    for (int a = 0; a < nDevs; ++a)
    {
        cudaCheck(cudaSetDevice(devs[a].device), "cudaSetDevice");
        for (int b = 0; b < nDevs; ++b)
        {
            if (a == b)
                continue;
            const cudaError_t err = cudaDeviceEnablePeerAccess(devs[b].device, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled)
                cudaGetLastError();
            else if (err != cudaSuccess)
            {
                cudaGetLastError();
                fprintf(stderr,
                        "[gpu] enabling peer access %d -> %d failed (%s): queries and hit flags "
                        "are staged through host memory\n",
                        devs[a].device, devs[b].device, cudaGetErrorString(err));
                return false;
            }
        }
    }
    fprintf(stderr,
            "[gpu] peer access on every pair: owners probe the encoder's partition in "
            "place (NVLink where present, PCIe P2P otherwise)\n");
    return true;
}

EncodeProbeEngine::~EncodeProbeEngine()
{
    for (int s = 0; s < nSlots && slots; ++s)
        if (slots[s].inFlight)
        {
            cudaSetDevice(devs[slots[s].encoder].device);
            cudaEventSynchronize(devs[slots[s].encoder].slots[s].evDone);
        }
    release();
}

void EncodeProbeEngine::release()
{
    if (slots)
    {
        for (int s = 0; s < nSlots; ++s)
        {
            cudaFreeHost(slots[s].h_seq);
            cudaFreeHost(slots[s].h_inv);
            cudaFreeHost(slots[s].h_desc);
            cudaFreeHost(slots[s].h_hit);
            cudaFreeHost(slots[s].h_valid);
        }
        delete[] slots;
        slots = nullptr;
    }
    if (devs)
    {
        for (int d = 0; d < nDevs; ++d)
        {
            cudaSetDevice(devs[d].device); // frees must target the owning device
            for (DevSlot &ds : devs[d].slots)
            {
                cudaFree(ds.d_seq);
                cudaFree(ds.d_inv);
                cudaFree(ds.d_desc);
                cudaFree(ds.d_hit);
                cudaFree(ds.d_valid);
                cudaFree(ds.d_codes);
                cudaFree(ds.d_partCodes);
                cudaFree(ds.d_partPos);
                cudaFree(ds.d_flags);
                cudaFree(ds.d_counts);
                cudaFree(ds.d_segOff);
                cudaFree(ds.d_cursor);
                cudaFree(ds.d_recvCodes);
                cudaFree(ds.d_recvSegOff);
                cudaFree(ds.d_localFlags);
                cudaFree(ds.d_codesAlt);
                cudaFree(ds.d_pos);
                cudaFree(ds.d_posAlt);
                cudaFree(ds.d_cubTemp);
                if (ds.encStream)
                    cudaStreamDestroy(ds.encStream);
                if (ds.ownStream)
                    cudaStreamDestroy(ds.ownStream);
                if (ds.evRouted)
                    cudaEventDestroy(ds.evRouted);
                if (ds.evProbed)
                    cudaEventDestroy(ds.evProbed);
                if (ds.evDone)
                    cudaEventDestroy(ds.evDone);
            }
        }
        delete[] devs;
        devs = nullptr;
    }
}

EncodeProbeEngine::HostSlot EncodeProbeEngine::host(int slot) const
{
    const Slot &sl = slots[slot];
    return {sl.h_seq, sl.h_inv, sl.h_desc, sl.h_hit, sl.h_valid, cap.outWords, (uint32_t)nDbs};
}

void EncodeProbeEngine::submit(int slot, uint32_t nWindows, size_t seqWordsUsed,
                               size_t invWordsUsed, size_t outWordsUsed)
{
    if (nWindows == 0 || outWordsUsed == 0)
        return; // nothing to probe; the slot stays reusable
    if (nDevs == 1)
        submitSingle(slot, nWindows, seqWordsUsed, invWordsUsed, outWordsUsed);
    else
        submitRouted(slot, nWindows, seqWordsUsed, invWordsUsed, outWordsUsed);
}

// One device: the Phase 2/3 pipeline, unchanged (fused kernel, or the
// sorted three-stage path).
void EncodeProbeEngine::submitSingle(int slot, uint32_t nWindows, size_t seqWordsUsed,
                                     size_t invWordsUsed, size_t outWordsUsed)
{
    Slot &sl = slots[slot];
    Dev &dv = devs[0];
    DevSlot &ds = dv.slots[slot];
    cudaCheck(cudaSetDevice(dv.device), "cudaSetDevice");
    cudaStream_t st = ds.encStream;

    cudaCheck(cudaMemcpyAsync(ds.d_seq, sl.h_seq, seqWordsUsed * 8, cudaMemcpyHostToDevice, st), "seq H2D");
    cudaCheck(cudaMemcpyAsync(ds.d_inv, sl.h_inv, invWordsUsed * 8, cudaMemcpyHostToDevice, st), "inv H2D");
    cudaCheck(cudaMemcpyAsync(ds.d_desc, sl.h_desc, nWindows * sizeof(WindowDesc), cudaMemcpyHostToDevice, st), "desc H2D");

    if (nDbs > 1)
    {
        // Panel mode: one encode, every database's probe, n hit masks.
        encodeProbeMultiKernel<<<gridFor(outWordsUsed, BLOCK), BLOCK, 0, st>>>(
            ds.d_desc, nWindows, ds.d_seq, ds.d_inv, outWordsUsed, k, dv.multi,
            ds.d_hit, cap.outWords, ds.d_valid);
        cudaCheck(cudaGetLastError(), "encode+probe (multi) kernel launch");
    }
    else if (!sorted)
    {
        const int grid = gridFor(outWordsUsed, BLOCK);
        withProbe(useHash, dv.hash, dv.bs, [&](auto probe)
        {
            encodeProbeKernel<<<grid, BLOCK, 0, st>>>(
                ds.d_desc, nWindows, ds.d_seq, ds.d_inv, outWordsUsed, k, probe,
                ds.d_hit, ds.d_valid);
        });
        cudaCheck(cudaGetLastError(), "encode+probe kernel launch");
    }
    else
    {
        const size_t P = outWordsUsed * 32;
        cudaCheck(cudaMemsetAsync(ds.d_codes, 0xFF, P * 8, st), "codes memset");
        encodeCollectKernel<<<gridFor(outWordsUsed, BLOCK), BLOCK, 0, st>>>(
            ds.d_desc, nWindows, ds.d_seq, ds.d_inv, outWordsUsed, k, ds.d_codes, ds.d_valid);
        cudaCheck(cudaGetLastError(), "encode-collect kernel launch");
        iotaKernel<<<gridFor(P, BLOCK), BLOCK, 0, st>>>(ds.d_pos, P);
        cudaCheck(cudaGetLastError(), "iota kernel launch");

        // Radix sort by k-mer value (low 2k bits suffice: canonical codes
        // never reach the all-ones value, so sentinels still sort last).
        cub::DoubleBuffer<uint64_t> keys(ds.d_codes, ds.d_codesAlt);
        cub::DoubleBuffer<uint32_t> vals(ds.d_pos, ds.d_posAlt);
        size_t tmp = dv.cubTempBytes;
        cudaCheck(cub::DeviceRadixSort::SortPairs(ds.d_cubTemp, tmp, keys, vals, P, 0,
                                                  2 * (int)k, st),
                  "radix sort");

        cudaCheck(cudaMemsetAsync(ds.d_hit, 0, outWordsUsed * 4, st), "hit clear");
        withProbe(useHash, dv.hash, dv.bs, [&](auto probe)
        {
            probeScatterKernel<<<gridFor(P, BLOCK), BLOCK, 0, st>>>(
                keys.Current(), vals.Current(), P, probe, ds.d_hit);
        });
        cudaCheck(cudaGetLastError(), "probe-scatter kernel launch");
    }

    for (int i = 0; i < nDbs; ++i)
        cudaCheck(cudaMemcpyAsync(sl.h_hit + (size_t)i * cap.outWords, ds.d_hit + (size_t)i * cap.outWords,
                                  outWordsUsed * 4, cudaMemcpyDeviceToHost, st),
                  "hit D2H");
    cudaCheck(cudaMemcpyAsync(sl.h_valid, ds.d_valid, outWordsUsed * 4, cudaMemcpyDeviceToHost, st), "valid D2H");
    cudaCheck(cudaEventRecord(ds.evDone, st), "event record");
    sl.encoder = 0;
    sl.inFlight = true;
}

// Several devices: one encoder per batch, queries routed to their shards.
void EncodeProbeEngine::submitRouted(int slot, uint32_t nWindows, size_t seqWordsUsed,
                                     size_t invWordsUsed, size_t outWordsUsed)
{
    Slot &sl = slots[slot];
    const int E = nextEncoder;
    nextEncoder = (nextEncoder + 1) % nDevs; // every device encodes 1/N of the batches
    Dev &enc = devs[E];
    DevSlot &es = enc.slots[slot];
    const size_t P = outWordsUsed * 32;
    // Each owner's flags: one shared block with peer access, else one block
    // per owner; a block is one lane per database (lane stride P).
    const size_t lane = direct ? 0 : maxPositions * (size_t)nDbs;

    OwnerByPrefix byPrefix{ownerShift, (uint32_t)nDevs, {}};
    for (int d = 0; d < nDevs; ++d)
        byPrefix.cutHi[d] = ownerCutHi[d];
    const OwnerByHash byHash{(uint32_t)nDevs};

    // --- encoder: H2D, K1 + owner count, offsets, partition ---------------
    cudaCheck(cudaSetDevice(enc.device), "cudaSetDevice");
    cudaCheck(cudaMemcpyAsync(es.d_seq, sl.h_seq, seqWordsUsed * 8, cudaMemcpyHostToDevice, es.encStream), "seq H2D");
    cudaCheck(cudaMemcpyAsync(es.d_inv, sl.h_inv, invWordsUsed * 8, cudaMemcpyHostToDevice, es.encStream), "inv H2D");
    cudaCheck(cudaMemcpyAsync(es.d_desc, sl.h_desc, nWindows * sizeof(WindowDesc), cudaMemcpyHostToDevice, es.encStream), "desc H2D");
    cudaCheck(cudaMemsetAsync(es.d_counts, 0, MAX_SHARDS * 4, es.encStream), "counts clear");
    for (int i = 0; i < nDbs; ++i)
        cudaCheck(cudaMemsetAsync(es.d_hit + (size_t)i * cap.outWords, 0, outWordsUsed * 4, es.encStream), "hit clear");
    // encodeWord emits only the positions inside each window, so the
    // padding entries at the tail of a window's last output word are never
    // written; pre-set every code to the sentinel (as the sorted path does)
    // or the partition would take stale codes of an earlier batch for
    // queries, overrun their owner's segment and overwrite the next
    // owner's queries (job 68741: hits lost, differently on each run).
    cudaCheck(cudaMemsetAsync(es.d_codes, 0xFF, P * 8, es.encStream), "codes memset");
    withOwner(useHash, byPrefix, byHash, [&](auto owner)
    {
        routeKernel<<<gridFor(outWordsUsed, BLOCK), BLOCK, 0, es.encStream>>>(
            es.d_desc, nWindows, es.d_seq, es.d_inv, outWordsUsed, k, owner,
            es.d_codes, es.d_valid, es.d_counts);
    });
    cudaCheck(cudaGetLastError(), "route kernel launch");
    segmentOffsetsKernel<<<1, 32, 0, es.encStream>>>(es.d_counts, nDevs, es.d_segOff, es.d_cursor);
    cudaCheck(cudaGetLastError(), "segment offsets kernel launch");
    withOwner(useHash, byPrefix, byHash, [&](auto owner)
    {
        partitionKernel<<<gridFor(P, BLOCK * PART_ITEMS), BLOCK, 0, es.encStream>>>(
            es.d_codes, P, owner, es.d_cursor, es.d_partCodes, es.d_partPos);
    });
    cudaCheck(cudaGetLastError(), "partition kernel launch");
    cudaCheck(cudaEventRecord(es.evRouted, es.encStream), "event record");

    // --- owners: probe own segment, flags back to the encoder --------------
    for (int o = 0; o < nDevs; ++o)
    {
        Dev &own = devs[o];
        DevSlot &os = own.slots[slot];
        cudaCheck(cudaSetDevice(own.device), "cudaSetDevice");
        cudaStream_t st = (o == E) ? es.encStream : os.ownStream;
        const uint64_t *codes = es.d_partCodes;
        const uint32_t *seg = es.d_segOff;
        uint8_t *flags = es.d_flags + (size_t)o * lane;
        if (o != E)
        {
            cudaCheck(cudaStreamWaitEvent(st, es.evRouted, 0), "wait for partition");
            if (!direct)
            {
                // No peer access: the whole partition (P codes; the segment
                // bounds are device-side) comes over as a staged copy.
                cudaCheck(cudaMemcpyPeerAsync(os.d_recvCodes, own.device, es.d_partCodes,
                                              enc.device, P * 8, st),
                          "staged codes copy");
                cudaCheck(cudaMemcpyPeerAsync(os.d_recvSegOff, own.device, es.d_segOff,
                                              enc.device, (nDevs + 1) * 4, st),
                          "staged offsets copy");
                codes = os.d_recvCodes;
                seg = os.d_recvSegOff;
                flags = os.d_localFlags;
            }
        }
        if (nDbs > 1)
            probeSegmentMultiKernel<<<gridFor(P, BLOCK), BLOCK, 0, st>>>(codes, seg, o, own.multi, flags, P);
        else
            withProbe(useHash, own.hash, own.bs, [&](auto probe)
            {
                probeSegmentKernel<<<gridFor(P, BLOCK), BLOCK, 0, st>>>(codes, seg, o, probe, flags);
            });
        cudaCheck(cudaGetLastError(), "probe-segment kernel launch");
        if (o != E)
        {
            if (!direct)
                cudaCheck(cudaMemcpyPeerAsync(es.d_flags + (size_t)o * lane, enc.device,
                                              os.d_localFlags, own.device, P * (size_t)nDbs, st),
                          "staged flags copy");
            cudaCheck(cudaEventRecord(os.evProbed, st), "event record");
        }
    }

    // --- encoder: assemble the hit mask, return both masks -----------------
    cudaCheck(cudaSetDevice(enc.device), "cudaSetDevice");
    for (int o = 0; o < nDevs; ++o)
        if (o != E)
            cudaCheck(cudaStreamWaitEvent(es.encStream, devs[o].slots[slot].evProbed, 0),
                      "wait for owner");
    if (nDbs > 1)
        scatterHitsMultiKernel<<<gridFor(P, BLOCK), BLOCK, 0, es.encStream>>>(
            es.d_flags, lane, P, (uint32_t)nDbs, es.d_partPos, es.d_segOff, nDevs, es.d_hit, cap.outWords);
    else
        scatterHitsKernel<<<gridFor(P, BLOCK), BLOCK, 0, es.encStream>>>(
            es.d_flags, lane, es.d_partPos, es.d_segOff, nDevs, es.d_hit);
    cudaCheck(cudaGetLastError(), "scatter-hits kernel launch");
    for (int i = 0; i < nDbs; ++i)
        cudaCheck(cudaMemcpyAsync(sl.h_hit + (size_t)i * cap.outWords, es.d_hit + (size_t)i * cap.outWords,
                                  outWordsUsed * 4, cudaMemcpyDeviceToHost, es.encStream),
                  "hit D2H");
    cudaCheck(cudaMemcpyAsync(sl.h_valid, es.d_valid, outWordsUsed * 4, cudaMemcpyDeviceToHost, es.encStream), "valid D2H");
    cudaCheck(cudaEventRecord(es.evDone, es.encStream), "event record");
    sl.encoder = E;
    sl.inFlight = true;
}

void EncodeProbeEngine::wait(int slot)
{
    Slot &sl = slots[slot];
    if (!sl.inFlight)
        return;
    // The encoder's completion event follows every owner's flags (its
    // stream waited on them), so one wait covers all devices.
    cudaCheck(cudaSetDevice(devs[sl.encoder].device), "cudaSetDevice");
    cudaCheck(cudaEventSynchronize(devs[sl.encoder].slots[slot].evDone), "event sync");
    sl.inFlight = false;
}
