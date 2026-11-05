#pragma once

// Host-side load of the KMC random-access index directly from the on-disk
// .kmc_pre / .kmc_suf files (documented KMC formats, versions KMC1 = 0 and
// KMC2 = 0x200), keeping the vendored API untouched. The two flat arrays
// (prefix LUTs + sorted suffix records) are reused verbatim; the suffix
// area is repacked during load to (a) strip the interleaved counter bytes
// (membership needs no counts) and (b) normalize the big-endian-by-byte
// records to native integers so a probe compares one word instead of
// looping over bytes. KMC2 additionally partitions k-mers into bins by
// minimizer signature; the signature map and the minimizer norm table are
// part of the probe structure and are replicated here so the search is
// bit-exact with CKMCFile::CheckKmer.
//
// This header is CUDA-free: it is the source of truth for what gets copied
// to the device, and its probe() is the CPU reference oracle for the GPU
// kernel (which mirrors it operation for operation).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

#include "ProbeCore.hpp"

// std::vector::resize value-initializes, i.e. zero-fills and first-touches
// the whole buffer on the calling thread. For a 16 GB suffix array that is
// several seconds of serial work ahead of a parallel repack that then
// overwrites every byte anyway. This allocator makes default construction
// a no-op, so the pages are first touched by the worker that fills them.
template <typename T, typename A = std::allocator<T>>
struct default_init_allocator : public A
{
    using A::A;
    template <typename U>
    struct rebind
    {
        using other = default_init_allocator<
            U, typename std::allocator_traits<A>::template rebind_alloc<U>>;
    };
    template <typename U>
    void construct(U *p) noexcept(std::is_nothrow_default_constructible<U>::value)
    {
        ::new (static_cast<void *>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U *p, Args &&...args)
    {
        std::allocator_traits<A>::construct(static_cast<A &>(*this), p,
                                            std::forward<Args>(args)...);
    }
};

class KmcRaIndex
{
public:
    using U32Vec = std::vector<uint32_t, default_init_allocator<uint32_t>>;
    using U64Vec = std::vector<uint64_t, default_init_allocator<uint64_t>>;

    // Fixed query-path parameters, shared with the device kernel.
    struct Params
    {
        uint32_t kmerLength = 0;
        uint32_t lutPrefixLength = 0;
        uint32_t signatureLen = 0;   // 0 => KMC1 (single LUT, no signatures)
        uint32_t suffixBytes = 0;    // (k - p) / 4
        uint64_t suffixMask = 0;     // low 2*(k - p) bits of a query
        uint64_t totalKmers = 0;
        uint64_t singleLutSize = 0;  // 4^p entries per bin
    };

    // The .kmc_pre header as stored (any k, any mode: nothing is checked
    // beyond the file's markers and format version).
    struct Header
    {
        uint32_t kmcVersion = 0;     // 0 (KMC1) or 0x200 (KMC2)
        uint32_t kmerLength = 0;
        uint32_t mode = 0;
        uint32_t counterSize = 0;
        uint32_t lutPrefixLength = 0;
        uint32_t signatureLen = 0;   // KMC2 only
        uint64_t totalKmers = 0;
        uint64_t headerOffset = 0;
        uint64_t fileBytes = 0;
    };

    // Reads only the header of <dbPathPrefix>.kmc_pre (a few bytes from
    // its end; no LUT), so a tool can choose its path before loading
    // anything: k (the k > 32 databases take the string path, which this
    // loader does not support) and the k-mer count (KDBIntersect sizes the
    // two databases by it). Throws on a missing or malformed file.
    static Header readHeader(const std::string &dbPathPrefix)
    {
        const std::string path = dbPathPrefix + ".kmc_pre";
        File file(path);
        if (!file.f)
            throw std::runtime_error("KmcRaIndex: cannot open " + path);
        return readHeaderOf(file.f, path);
    }

    fastibs::ProbeParams probeParams() const
    {
        return {params.kmerLength, params.lutPrefixLength, params.signatureLen,
                params.suffixMask,  params.totalKmers,     params.singleLutSize};
    }

    // lutOnly: read the .kmc_pre (LUT, signature map, header) and skip the
    // suffix records — enough for the parameters, the per-prefix counts
    // (a shard plan) and the device footprint, without the suffix array in
    // host RAM; probe() must not be called on such an instance.
    explicit KmcRaIndex(const std::string &dbPathPrefix, bool lutOnly = false)
    {
        const auto t0 = std::chrono::steady_clock::now();
        loadPre(dbPathPrefix + ".kmc_pre");
        if (!lutOnly)
            loadSuf(dbPathPrefix + ".kmc_suf");
        if (params.signatureLen)
            buildNorm(params.signatureLen);
        loadSecs = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    }

    // Wall time of the on-disk read + repack, the thread count used for
    // the suffix repack (FASTIBS_LOAD_THREADS overrides the default, which
    // is the number of CPUs in this process's affinity mask; 0, empty or
    // non-numeric keeps the default; the count is clamped to 1..64 and
    // files under 4M records load single-threaded), and the
    // bytes of suffix records read — so callers can print GB/s and tell a
    // cold read from storage apart from a warm one from the page cache.
    double loadSeconds() const { return loadSecs; }
    unsigned loadThreads() const { return nLoadThreads; }
    uint64_t loadBytes() const { return loadBytesRead; }

    const Params &getParams() const { return params; }
    const std::vector<uint64_t> &getLut() const { return lut; }
    const std::vector<uint32_t> &getSignatureMap() const { return signatureMap; }
    const std::vector<uint32_t> &getNorm() const { return norm; }
    // Counter-stripped suffix records, one native integer per record:
    // suffix32 when the record's suffix fits 4 bytes, else suffix64.
    bool recordsAre32Bit() const { return params.suffixBytes <= 4; }
    const U32Vec &getSuffix32() const { return suffix32; }
    const U64Vec &getSuffix64() const { return suffix64; }

    size_t deviceFootprintBytes() const
    {
        return lut.size() * 8 + signatureMap.size() * 4 + norm.size() * 4 +
               suffix32.size() * 4 + suffix64.size() * 8;
    }

    // Enumerates every k-mer of the database as its canonical 2-bit packed
    // code, reconstructed from the index itself: consecutive LUT entries
    // bound each (bin, prefix) bucket, whose records share the prefix
    // encoded by the entry's position within its bin. No KMC listing pass
    // is needed. Order is the on-disk record order.
    template <typename Fn>
    void forEachKmer(Fn &&fn) const
    {
        forEachKmerIn(0, numBuckets(), fn);
    }

    // (bin, prefix) buckets of the LUT: its entries minus the guard.
    size_t numBuckets() const { return lut.empty() ? 0 : lut.size() - 1; }

    // CPUs this process may actually run on. hardware_concurrency() is the
    // machine's online CPU count, which under a SLURM/cgroup cpuset can be
    // far above the allocation (64 reported vs 16 allotted on the
    // evaluation cluster, job 58561); the affinity mask is the honest
    // number, and it also keeps a cold read from being fanned out over 64
    // concurrent streams on a shared filesystem. The default thread count
    // of the suffix repack and of KDBIntersect's enumeration.
    static unsigned usableCpus()
    {
#ifdef __linux__
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0)
        {
            const int n = CPU_COUNT(&set);
            if (n > 0)
                return (unsigned)n;
        }
#endif
        const unsigned hc = std::thread::hardware_concurrency();
        return hc ? hc : 1u;
    }

    // The same walk over the buckets [jBegin, jEnd) only — the unit of the
    // parallel enumeration (ParallelEnum.hpp): the buckets partition the
    // records, so walking disjoint bucket ranges on several threads visits
    // every k-mer exactly once, each reconstructed as forEachKmer does.
    template <typename Fn>
    void forEachKmerIn(size_t jBegin, size_t jEnd, Fn &&fn) const
    {
        jEnd = std::min(jEnd, numBuckets());
        if (recordsAre32Bit())
            walkBuckets(suffix32.data(), jBegin, jEnd, fn);
        else
            walkBuckets(suffix64.data(), jBegin, jEnd, fn);
    }

    // Minimizer signature of a canonical k-mer given as a 2-bit packed code.
    // Delegates to the shared host/device implementation in ProbeCore.hpp.
    uint32_t signatureOf(uint64_t code) const
    {
        return fastibs::signatureOf(code, params.kmerLength, params.signatureLen,
                                    norm.data());
    }

    // Membership probe mirroring CKMCFile::CheckKmer. Delegates to the
    // shared host/device implementation in ProbeCore.hpp — the same
    // template every CUDA kernel calls, so the host result validated by
    // tests/phase1_test.cpp is exactly what the device computes.
    bool probe(uint64_t code) const
    {
        const auto P = probeParams();
        if (recordsAre32Bit())
            return fastibs::probeOne<uint32_t>(code, lut.data(), suffix32.data(),
                                               signatureMap.data(), norm.data(), P);
        return fastibs::probeOne<uint64_t>(code, lut.data(), suffix64.data(),
                                           signatureMap.data(), norm.data(), P);
    }

    // ---- Splitting the index across devices (Phase 5) --------------------
    //
    // A shard is a contiguous range [prefixLo, prefixHi) of the 4^p prefix
    // buckets, taken from EVERY signature bin. Within a bin the records are
    // sorted by prefix, so a shard's records are one contiguous run per bin;
    // shardLayout gathers those runs bin after bin into one array and
    // rebases the bins' LUT entries of the range onto it. probeOne routes on
    // the query's prefix — known right after the encode, before the
    // signature scan — so a device rejects the other shards' queries almost
    // for free, and every k-mer is held by exactly one shard.
    //
    // The plan balances RECORD counts, not prefixes: canonical k-mers are
    // skewed towards small prefixes (canonical = min of the two strands, so
    // 'A'-leading codes dominate), hence a cumulative cut of the per-prefix
    // record counts. Balance is within one prefix bucket's share.
    struct Shard
    {
        uint32_t prefixLo = 0, prefixHi = 0; // [lo, hi) of the 4^p prefixes
        uint64_t records = 0;                // records held by the shard
        int index = 0, count = 1;            // shard number / number of shards
    };

    // Signature bins in the LUT (1 for KMC1).
    uint32_t numBins() const
    {
        return (uint32_t)((lut.size() - 1) / params.singleLutSize);
    }

    // Records per prefix bucket, summed over the signature bins (4^p
    // entries): what the plan balances. Needs the LUT only.
    std::vector<uint64_t> perPrefixCounts() const
    {
        const uint64_t P4 = params.singleLutSize;
        const uint64_t total = params.totalKmers;
        const uint32_t bins = numBins();

        std::vector<uint64_t> perPrefix(P4, 0);
        for (uint32_t b = 0; b < bins; ++b)
        {
            const uint64_t base = (uint64_t)b * P4;
            for (uint64_t q = 0; q < P4; ++q)
            {
                const uint64_t lo = lut[base + q];
                const uint64_t hi = std::min<uint64_t>(lut[base + q + 1], total);
                if (hi > lo)
                    perPrefix[q] += hi - lo;
            }
        }
        return perPrefix;
    }

    std::vector<Shard> shardPlan(int nShards) const
    {
        return planFromCounts(perPrefixCounts(), nShards);
    }

    // One cut for several databases (panel mode on several devices,
    // docs/MULTIDB_SCAN.md section 3): the per-prefix counts of every
    // database, each over its own 4^p_i buckets, are folded to the
    // coarsest prefix length p_min among them (4^(p_i - p_min)
    // consecutive buckets of database i are one p_min bucket), summed, and
    // cut by record count like a single database's plan. Database i's
    // shard d is the plan's range scaled by 4^(p_i - p_min): the same set
    // of k-mers' prefixes for every database, so the device-side owner
    // function (prefix >> 2(k - p_min) against the common cuts) is one
    // function for all of them. Returns plan[i][d].
    static std::vector<std::vector<Shard>>
    jointShardPlan(const std::vector<std::vector<uint64_t>> &perPrefix, int nShards)
    {
        if (perPrefix.empty())
            return {};
        size_t pMinLen = perPrefix[0].size();
        for (const auto &c : perPrefix)
            pMinLen = std::min(pMinLen, c.size());
        std::vector<uint64_t> joint(pMinLen, 0);
        for (const auto &c : perPrefix)
        {
            const size_t fold = c.size() / pMinLen; // 4^(p_i - p_min)
            for (size_t q = 0; q < c.size(); ++q)
                joint[q / fold] += c[q];
        }
        const std::vector<Shard> common = planFromCounts(joint, nShards);
        std::vector<std::vector<Shard>> plan(perPrefix.size());
        for (size_t i = 0; i < perPrefix.size(); ++i)
        {
            const uint64_t scale = perPrefix[i].size() / pMinLen;
            plan[i] = common;
            for (Shard &s : plan[i])
            {
                s.prefixLo = (uint32_t)(s.prefixLo * scale);
                s.prefixHi = (uint32_t)(s.prefixHi * scale);
                s.records = 0;
                for (uint64_t q = s.prefixLo; q < s.prefixHi; ++q)
                    s.records += perPrefix[i][q];
            }
        }
        return plan;
    }

    static std::vector<Shard> planFromCounts(const std::vector<uint64_t> &perPrefix,
                                             int nShards)
    {
        const uint64_t P4 = perPrefix.size();
        uint64_t total = 0;
        for (uint64_t c : perPrefix)
            total += c;

        std::vector<Shard> plan((size_t)std::max(1, nShards));
        uint64_t acc = 0, q = 0;
        for (size_t d = 0; d < plan.size(); ++d)
        {
            Shard &s = plan[d];
            s.index = (int)d;
            s.count = (int)plan.size();
            s.prefixLo = (uint32_t)q;
            const uint64_t before = acc;
            if (d + 1 == plan.size())
            {
                while (q < P4) // the last shard takes what is left
                    acc += perPrefix[q++];
            }
            else
            {
                // Advance the cut while it brings the cumulative count
                // closer to this shard's ideal share.
                const uint64_t target = total * (uint64_t)(d + 1) / plan.size();
                while (q < P4)
                {
                    const uint64_t next = acc + perPrefix[q];
                    const uint64_t dNext = next > target ? next - target : target - next;
                    const uint64_t dCur = acc > target ? acc - target : target - acc;
                    if (dNext > dCur)
                        break;
                    acc = next;
                    ++q;
                }
            }
            s.prefixHi = (uint32_t)q;
            s.records = acc - before;
        }
        return plan;
    }

    // The device image of a shard: its rebased LUT and, per bin in gather
    // order, the global record range to copy. Consecutive bins share their
    // boundary LUT entry only when the shard spans every prefix (bin b's end
    // is bin b+1's start) — that stride is the on-disk layout, so the single
    // shard of a one-device run reproduces the on-disk arrays exactly. A
    // partial range needs prefixHi - prefixLo + 1 entries per bin.
    struct ShardLayout
    {
        std::vector<uint64_t> lut;
        std::vector<std::pair<uint64_t, uint64_t>> ranges; // per bin: [lo, hi)
        uint64_t records = 0;
        fastibs::ProbeParams params{};
    };

    ShardLayout shardLayout(const Shard &s) const
    {
        const uint64_t P4 = params.singleLutSize;
        const uint64_t total = params.totalKmers;
        const uint32_t bins = numBins();
        const uint64_t w = s.prefixHi - s.prefixLo;
        const uint64_t stride = w == P4 ? P4 : w + 1;

        ShardLayout out;
        out.lut.resize((bins - 1) * stride + w + 1);
        out.ranges.reserve(bins);
        for (uint32_t b = 0; b < bins; ++b)
        {
            const uint64_t base = (uint64_t)b * P4;
            const uint64_t binLo = lut[base + s.prefixLo];
            const uint64_t binHi = std::max(
                binLo, std::min<uint64_t>(lut[base + s.prefixHi], total));
            // Rebased without clamping, so the guard entry past the last
            // bucket keeps KMC's total+1 quirk (probeOne clamps it).
            for (uint64_t i = 0; i <= w; ++i)
                out.lut[b * stride + i] = lut[base + s.prefixLo + i] - binLo + out.records;
            out.ranges.emplace_back(binLo, binHi);
            out.records += binHi - binLo;
        }
        out.params = probeParams();
        out.params.totalKmers = out.records;
        out.params.singleLutSize = stride;
        out.params.prefixLo = s.prefixLo;
        out.params.prefixHi = s.prefixHi;
        return out;
    }

private:
    Params params;
    std::vector<uint64_t> lut;          // num_bins * 4^p + 1 entries
    std::vector<uint32_t> signatureMap; // 4^signatureLen + 1 entries (KMC2)
    std::vector<uint32_t> norm;         // 4^signatureLen entries (KMC2)
    U32Vec suffix32; // default-init: filled (and first-touched) by the repack workers
    U64Vec suffix64;

    // The record walk behind forEachKmer / forEachKmerIn, with the record
    // width resolved once outside the loops. singleLutSize is 4^p, so the
    // position within the bin is a mask of the bucket index.
    template <typename Rec, typename Fn>
    void walkBuckets(const Rec *records, size_t jBegin, size_t jEnd, Fn &fn) const
    {
        const uint32_t shift = 2 * (params.kmerLength - params.lutPrefixLength);
        const uint64_t prefixMask = params.singleLutSize - 1;
        const uint64_t total = params.totalKmers;
        for (size_t j = jBegin; j < jEnd; ++j)
        {
            const uint64_t prefix = ((uint64_t)j & prefixMask) << shift;
            const uint64_t lo = lut[j];
            const uint64_t hi = std::min<uint64_t>(lut[j + 1], total);
            for (uint64_t r = lo; r < hi; ++r)
                fn(prefix | (uint64_t)records[r]);
        }
    }

    struct File
    {
        FILE *f = nullptr;
        explicit File(const std::string &path) { f = fopen(path.c_str(), "rb"); }
        ~File()
        {
            if (f)
                fclose(f);
        }
    };

    static void readAt(FILE *f, long long offset, void *dst, size_t bytes, const char *what)
    {
#ifdef _WIN32
        _fseeki64(f, offset, SEEK_SET);
#else
        fseeko(f, offset, SEEK_SET);
#endif
        if (fread(dst, 1, bytes, f) != bytes)
            throw std::runtime_error(std::string("KmcRaIndex: short read of ") + what);
    }

    static uint64_t fileSize(FILE *f)
    {
#ifdef _WIN32
        _fseeki64(f, 0, SEEK_END);
        return (uint64_t)_ftelli64(f);
#else
        fseeko(f, 0, SEEK_END);
        return (uint64_t)ftello(f);
#endif
    }

    static void checkMarkers(FILE *f, uint64_t size, const char *marker, const char *path)
    {
        char head[4], tail[4];
        readAt(f, 0, head, 4, "marker");
        readAt(f, (long long)size - 4, tail, 4, "marker");
        if (memcmp(head, marker, 4) != 0 || memcmp(tail, marker, 4) != 0)
            throw std::runtime_error(std::string("KmcRaIndex: bad marker in ") + path);
    }

    // The header fields of an open .kmc_pre (the markers checked, nothing
    // else read): shared by loadPre and readHeader.
    static Header readHeaderOf(FILE *f, const std::string &path)
    {
        const uint64_t fsize = fileSize(f);
        checkMarkers(f, fsize, "KMCP", path.c_str());

        Header hd;
        hd.fileBytes = fsize;
        readAt(f, (long long)fsize - 12, &hd.kmcVersion, 4, "version");
        if (hd.kmcVersion != 0 && hd.kmcVersion != 0x200)
            throw std::runtime_error("KmcRaIndex: unsupported KMC format version");

        // header_offset is stored as 4 bytes before the end marker; KMC
        // reads only its first byte (fgetc), which we replicate.
        unsigned char headerOffsetByte;
        readAt(f, (long long)fsize - 8, &headerOffsetByte, 1, "header offset");
        hd.headerOffset = headerOffsetByte;

        const long long headerPos = (long long)fsize - (long long)(hd.headerOffset + 8);

        if (hd.kmcVersion == 0x200)
        {
            uint32_t h[5];
            readAt(f, headerPos, h, sizeof(h), "header");
            hd.kmerLength = h[0];
            hd.mode = h[1];
            hd.counterSize = h[2];
            hd.lutPrefixLength = h[3];
            hd.signatureLen = h[4];
            readAt(f, headerPos + 7 * 4, &hd.totalKmers, 8, "total_kmers");
        }
        else
        {
            uint32_t h[4];
            readAt(f, headerPos, h, sizeof(h), "header");
            hd.kmerLength = h[0];
            hd.mode = h[1];
            hd.counterSize = h[2];
            hd.lutPrefixLength = h[3];
            readAt(f, headerPos + 6 * 4, &hd.totalKmers, 8, "total_kmers");
        }
        return hd;
    }

    void loadPre(const std::string &path)
    {
        File file(path);
        if (!file.f)
            throw std::runtime_error("KmcRaIndex: cannot open " + path);
        const Header hd = readHeaderOf(file.f, path);
        const uint64_t fsize = hd.fileBytes;
        const uint32_t kmcVersion = hd.kmcVersion;
        const uint64_t headerOffset = hd.headerOffset;
        const uint32_t kmerLength = hd.kmerLength, mode = hd.mode,
                       counterSize = hd.counterSize, lutPrefixLength = hd.lutPrefixLength,
                       signatureLen = hd.signatureLen;
        const uint64_t totalKmers = hd.totalKmers;
        if (mode != 0)
            throw std::runtime_error("KmcRaIndex: Quake-compatible counters are not supported");
        if (kmerLength > 32)
            throw std::runtime_error("KmcRaIndex: k > 32 is not supported");
        if ((kmerLength - lutPrefixLength) % 4 != 0 || kmerLength - lutPrefixLength > 32)
            throw std::runtime_error("KmcRaIndex: unexpected suffix geometry");

        params.kmerLength = kmerLength;
        params.lutPrefixLength = lutPrefixLength;
        params.signatureLen = (kmcVersion == 0x200) ? signatureLen : 0;
        params.suffixBytes = (kmerLength - lutPrefixLength) / 4;
        const uint32_t suffixSymbols = kmerLength - lutPrefixLength;
        params.suffixMask =
            suffixSymbols == 32 ? ~0ull : (1ull << (2 * suffixSymbols)) - 1;
        params.totalKmers = totalKmers;
        params.singleLutSize = 1ull << (2 * lutPrefixLength);
        counterBytes = counterSize;

        // LUT area: same arithmetic as CKMCFile::ReadParamsFrom_prefix_file_buf.
        if (kmcVersion == 0x200)
        {
            const uint64_t sigMapEntries = (1ull << (2 * signatureLen)) + 1;
            const uint64_t sizeNoMarkers = fsize - 8 - 4; // minus header_offset field
            const uint64_t lutAreaBytes =
                sizeNoMarkers - (sigMapEntries * 4 + headerOffset + 8);
            const uint64_t lutEntries = (lutAreaBytes + 8) / 8;
            lut.resize(lutEntries);
            readAt(file.f, 4, lut.data(), lutAreaBytes + 8, "prefix LUT");
            lut[lutEntries - 1] = totalKmers + 1;

            signatureMap.resize(sigMapEntries);
            readAt(file.f, 4 + (long long)lutAreaBytes + 8, signatureMap.data(),
                   sigMapEntries * 4, "signature map");
        }
        else
        {
            const uint64_t lutEntries = params.singleLutSize + 1;
            lut.resize(lutEntries);
            readAt(file.f, 4, lut.data(), lutEntries * 8, "prefix LUT");
            lut[lutEntries - 1] = totalKmers + 1;
        }
    }

    void loadSuf(const std::string &path)
    {
        File file(path);
        if (!file.f)
            throw std::runtime_error("KmcRaIndex: cannot open " + path);
        const uint64_t fsize = fileSize(file.f);
        checkMarkers(file.f, fsize, "KMCS", path.c_str());

        const uint32_t recBytes = params.suffixBytes + counterBytes;
        const uint64_t dataBytes = fsize - 8;
        if (dataBytes < params.totalKmers * (uint64_t)recBytes)
            throw std::runtime_error("KmcRaIndex: suffix file smaller than expected");

        // Stream records, strip counters, normalize the big-endian-by-byte
        // suffix to a native integer (numeric order == KMC's byte order).
        //
        // The repack is CPU-bound (a byte loop per record; for the 2e9-
        // record production database it cost roughly 12-16 s single-threaded,
        // which made the GPU path load slower than the CPU path's plain file
        // read). Records are fixed-size and the destination is pre-sized,
        // so the range is split into contiguous slices, each read and
        // repacked by its own thread through its own file handle into a
        // disjoint part of the output — no synchronization needed.
        const bool use32 = recordsAre32Bit();
        if (use32)
            suffix32.resize(params.totalKmers);
        else
            suffix64.resize(params.totalKmers);

        unsigned threads = usableCpus();
        if (const char *env = std::getenv("FASTIBS_LOAD_THREADS"))
        {
            const int n = std::atoi(env); // 0, empty or non-numeric: keep the default
            if (n > 0)
                threads = (unsigned)n;
        }
        threads = std::max(1u, std::min(threads, 64u));
        if (params.totalKmers < (4u << 20)) // not worth spawning for small files
            threads = 1;
        nLoadThreads = threads;
        loadBytesRead = params.totalKmers * (uint64_t)recBytes;

        const uint64_t total = params.totalKmers;
        const uint32_t suffixBytes = params.suffixBytes;
        uint32_t *out32 = use32 ? suffix32.data() : nullptr;
        uint64_t *out64 = use32 ? nullptr : suffix64.data();

        auto repackRange = [&](uint64_t first, uint64_t last)
        {
            File f(path);
            if (!f.f)
                throw std::runtime_error("KmcRaIndex: cannot open " + path);
#ifdef _WIN32
            _fseeki64(f.f, 4 + (long long)(first * recBytes), SEEK_SET);
#else
            fseeko(f.f, 4 + (off_t)(first * recBytes), SEEK_SET);
#endif
            std::vector<unsigned char> buf((size_t)1 << 20);
            const size_t recsPerChunk = buf.size() / recBytes;
            uint64_t rec = first;
            while (rec < last)
            {
                const size_t batch =
                    (size_t)std::min<uint64_t>(recsPerChunk, last - rec);
                if (fread(buf.data(), 1, batch * recBytes, f.f) != batch * recBytes)
                    throw std::runtime_error("KmcRaIndex: short read of suffix records");
                const unsigned char *q = buf.data();
                for (size_t i = 0; i < batch; ++i, q += recBytes)
                {
                    uint64_t v = 0;
                    for (uint32_t b = 0; b < suffixBytes; ++b)
                        v = (v << 8) | q[b];
                    if (use32)
                        out32[rec + i] = (uint32_t)v;
                    else
                        out64[rec + i] = v;
                }
                rec += batch;
            }
        };

        if (threads == 1)
        {
            repackRange(0, total);
            return;
        }

        std::vector<std::thread> workers;
        std::vector<std::exception_ptr> errors(threads);
        const uint64_t per = (total + threads - 1) / threads;
        for (unsigned t = 0; t < threads; ++t)
        {
            const uint64_t first = std::min<uint64_t>(total, (uint64_t)t * per);
            const uint64_t last = std::min<uint64_t>(total, first + per);
            workers.emplace_back([&, t, first, last]
            {
                try
                {
                    if (first < last)
                        repackRange(first, last);
                }
                catch (...)
                {
                    errors[t] = std::current_exception();
                }
            });
        }
        for (auto &w : workers)
            w.join();
        for (auto &e : errors)
            if (e)
                std::rethrow_exception(e);
    }

    // Minimizer norm table, replicated from the vendored CMmer (mmer.h):
    // norm[m] = min(m if allowed else 4^len, rev(m) if allowed else 4^len).
    static bool mmerAllowed(uint32_t mmer, uint32_t len)
    {
        if ((mmer & 0x3f) == 0x3f) // TTT suffix
            return false;
        if ((mmer & 0x3f) == 0x3b) // TGT suffix
            return false;
        if ((mmer & 0x3c) == 0x3c) // TG* suffix
            return false;
        for (uint32_t j = 0; j < len - 3; ++j)
            if ((mmer & 0xf) == 0) // AA inside
                return false;
            else
                mmer >>= 2;
        if (mmer == 0) // AAA prefix
            return false;
        if (mmer == 0x04) // ACA prefix
            return false;
        if ((mmer & 0xf) == 0) // *AA prefix
            return false;
        return true;
    }

    void buildNorm(uint32_t len)
    {
        const uint32_t special = 1u << (2 * len);
        norm.resize(special);
        for (uint32_t i = 0; i < special; ++i)
        {
            uint32_t rev = 0, m = i;
            for (uint32_t j = 0; j < len; ++j)
            {
                rev = (rev << 2) | (3 - (m & 3));
                m >>= 2;
            }
            const uint32_t strVal = mmerAllowed(i, len) ? i : special;
            const uint32_t revVal = mmerAllowed(rev, len) ? rev : special;
            norm[i] = strVal < revVal ? strVal : revVal;
        }
    }

    uint32_t counterBytes = 0;
    double loadSecs = 0;
    unsigned nLoadThreads = 1;
    uint64_t loadBytesRead = 0;
};
