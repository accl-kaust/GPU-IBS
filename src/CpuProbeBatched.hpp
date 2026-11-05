#pragma once

// The batched CPU probe (CPU-BATCHED): K membership probes kept in flight
// per worker thread over the host image of the KMC index (KmcRaIndex — the
// same counter-stripped, native-integer arrays the GPU path uploads).
//
// CPU-OPT probes one canonical k-mer at a time, so every step of the
// binary search over a (bin, prefix) bucket stalls on one DRAM miss and a
// core keeps a single miss outstanding. The GPU wins this workload mostly
// by having thousands of probes in flight; this is the CPU counterpart, the
// baseline a reviewer would ask for. A worker gathers K canonical codes as
// two independent groups of K/2 and runs their probes level by level: the
// minimizer signatures, then the signature-map entries, then the pairs of
// LUT entries, then at every binary-search step the next suffix records —
// each level prefetched (__builtin_prefetch) for a whole group before any
// of it is read, so K independent cache misses overlap instead of
// serializing. The two groups are software-pipelined against each other:
// one group's records are in flight while the other group's step runs, so
// the memory system is never idle while a core computes. The search step
// is branchless (the compare feeds conditional moves, not a branch, so a
// mispredicted hit/miss cannot drain the loads in flight). The arithmetic
// is probeOne's (gpu/ProbeCore.hpp) operation for operation, so the answer
// is bit-identical to the GPU kernel's and to CKMCFile::IsKmer
// (tests/phase1_test.cpp checks all three against each other). Host-only:
// prefetch has no meaning in the device kernels.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "gpu/KmcRaIndex.hpp"
#include "gpu/ProbeCore.hpp"

namespace fastibs
{

// Upper bound on K: the per-thread working set is a few arrays of K words
// on the stack, and beyond a few dozen outstanding misses a core's miss
// queue, not K, is the limit.
constexpr unsigned kCpuBatchMax = 256;

// Read prefetch into every cache level; the line is used right after the
// level's prefetch pass. A no-op on compilers without the builtin (the
// result does not depend on it).
inline void prefetchRead(const void *p)
{
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0, 3);
#else
    (void)p;
#endif
}

// One group of up to kCpuBatchMax probes advanced level by level. Slots
// that drop out at any level are compacted away; hits are written by
// position (idx is the position in the caller's codes[], size_t so any
// chunk length is fine), so the slot order is free.
template <typename RecT>
struct ProbeGroup
{
    size_t idx[kCpuBatchMax];     // position in codes[] held by each slot
    uint64_t entry[kCpuBatchMax]; // the slot's LUT entry: bin base + prefix
    uint32_t sig[kCpuBatchMax];   // minimizer signature (KMC2)
    long long lo[kCpuBatchMax], hi[kCpuBatchMax], mid[kCpuBatchMax];
    RecT target[kCpuBatchMax];    // the suffix the bucket is searched for
    uint8_t hit[kCpuBatchMax], done[kCpuBatchMax];
    unsigned count = 0; // slots in use at the current level

    // Gathers up to cap queries from codes[i..n), advancing i; masked
    // positions are misses by definition.
    void gather(const uint8_t *valid, size_t &i, size_t n, uint8_t *hits, unsigned cap)
    {
        count = 0;
        for (; count < cap && i < n; ++i)
        {
            if (valid && !valid[i])
                hits[i] = 0;
            else
                idx[count++] = i;
        }
    }

    // Level 1: route on the prefix (another shard's query is a miss,
    // decided before the signature scan as in probeOne), compute the
    // signature, start the signature-map fetches.
    void routeAndSign(const uint64_t *codes, const uint32_t *norm, const uint32_t *sigMap,
                      const ProbeParams &P, uint32_t prefixShift, uint8_t *hits)
    {
        unsigned live = 0;
        for (unsigned j = 0; j < count; ++j)
        {
            const uint64_t code = codes[idx[j]];
            const uint64_t prefix = code >> prefixShift;
            if (prefix < P.prefixLo || prefix >= P.prefixHi)
            {
                hits[idx[j]] = 0;
                continue;
            }
            idx[live] = idx[j];
            entry[live] = prefix - P.prefixLo;
            if (P.sigLen)
            {
                sig[live] = signatureOf(code, P.k, P.sigLen, norm);
                prefetchRead(sigMap + sig[live]);
            }
            ++live;
        }
        count = live;
    }

    // Level 2: the bin's LUT base from the signature map, then start the
    // pairs of LUT fetches (the bucket's two bounds; usually one cache
    // line, two when the entry sits at a line end).
    void lutPrefetch(const uint64_t *lut, const uint32_t *sigMap, const ProbeParams &P)
    {
        for (unsigned j = 0; j < count; ++j)
        {
            if (P.sigLen)
                entry[j] += (uint64_t)sigMap[sig[j]] * P.singleLutSize;
            prefetchRead(lut + entry[j]);
            prefetchRead(lut + entry[j] + 1);
        }
    }

    // Level 3: bucket bounds and the compare target. KMC's guard entry
    // past the last bucket is total+1 (probeOne clamps it the same way);
    // an empty bucket is a miss before any record is touched.
    void bounds(const uint64_t *codes, const uint64_t *lut, const ProbeParams &P, uint8_t *hits)
    {
        const long long last = (long long)P.totalKmers - 1;
        unsigned active = 0;
        for (unsigned j = 0; j < count; ++j)
        {
            const long long l = (long long)lut[entry[j]];
            long long h = (long long)lut[entry[j] + 1] - 1;
            if (l >= (long long)P.totalKmers)
            {
                hits[idx[j]] = 0;
                continue;
            }
            if (h > last)
                h = last;
            if (l > h)
            {
                hits[idx[j]] = 0;
                continue;
            }
            idx[active] = idx[j];
            lo[active] = l;
            hi[active] = h;
            target[active] = (RecT)(codes[idx[j]] & P.suffixMask);
            ++active;
        }
        count = active;
    }

    // Level 4, first half of a step: every slot's next record goes in
    // flight.
    void prefetchStep(const RecT *suffixes)
    {
        for (unsigned j = 0; j < count; ++j)
        {
            mid[j] = (lo[j] + hi[j]) >> 1;
            prefetchRead(suffixes + mid[j]);
        }
    }

    // Level 4, second half: read the record and step, branchlessly — the
    // compare selects the new bound (probeOne: lo = mid + 1 when the
    // record is below the target, else hi = mid - 1; equal is a hit). A
    // slot leaves when it hits or its range empties; the compaction pass
    // runs only when some slot did.
    void processStep(const RecT *suffixes, uint8_t *hits)
    {
        unsigned nDone = 0;
        for (unsigned j = 0; j < count; ++j)
        {
            const RecT s = suffixes[mid[j]];
            const bool lt = s < target[j];
            const bool eq = s == target[j];
            lo[j] = lt ? mid[j] + 1 : lo[j];
            hi[j] = lt ? hi[j] : mid[j] - 1;
            hit[j] = eq;
            done[j] = eq | (lo[j] > hi[j]);
            nDone += done[j];
        }
        if (!nDone)
            return;
        unsigned keep = 0;
        for (unsigned j = 0; j < count; ++j)
        {
            if (done[j])
            {
                hits[idx[j]] = hit[j];
                continue;
            }
            idx[keep] = idx[j];
            lo[keep] = lo[j];
            hi[keep] = hi[j];
            target[keep] = target[j];
            ++keep;
        }
        count = keep;
    }
};

// Probes codes[0..n) against one probe structure (the whole index, or a
// shard: queries outside its prefix range are misses, as in probeOne) and
// writes hits[i] = 0/1. `valid` (nullable) masks positions whose code is
// unspecified — the mapper's per-position encode; a masked position is a
// miss and costs nothing. K probes are kept in flight per call, as two
// groups of K/2 (A takes the odd one; K = 1 is the plain probe with
// prefetches: B stays empty and every call on it is a no-op).
template <typename RecT>
inline void probeBatched(const uint64_t *codes, const uint8_t *valid, size_t n,
                         uint8_t *hits, const uint64_t *lut, const RecT *suffixes,
                         const uint32_t *sigMap, const uint32_t *norm,
                         const ProbeParams &P, unsigned K)
{
    K = std::max(1u, std::min(K, kCpuBatchMax));
    const unsigned capA = (K + 1) / 2, capB = K / 2;
    const uint32_t prefixShift = 2 * (P.k - P.p);
    ProbeGroup<RecT> A, B;

    size_t i = 0;
    while (i < n)
    {
        A.gather(valid, i, n, hits, capA);
        B.gather(valid, i, n, hits, capB);
        // Levels 1-3 alternate the groups too: B's signatures are computed
        // while A's signature-map lines are in flight, and so on.
        A.routeAndSign(codes, norm, sigMap, P, prefixShift, hits);
        B.routeAndSign(codes, norm, sigMap, P, prefixShift, hits);
        A.lutPrefetch(lut, sigMap, P);
        B.lutPrefetch(lut, sigMap, P);
        A.bounds(codes, lut, P, hits);
        B.bounds(codes, lut, P, hits);
        // Level 4, software-pipelined across the groups: A's step t reads
        // what was prefetched before B's step t-1 ran, then prefetches
        // step t+1 for A while B's step t runs — one group's records are
        // always in flight. A group whose searches are all done is a
        // no-op until the other finishes; then the next K are gathered.
        A.prefetchStep(suffixes);
        B.prefetchStep(suffixes);
        while (A.count | B.count)
        {
            A.processStep(suffixes, hits);
            A.prefetchStep(suffixes);
            B.processStep(suffixes, hits);
            B.prefetchStep(suffixes);
        }
    }
}

// The same over a host index image (whole index: every prefix is ours).
inline void probeBatched(const KmcRaIndex &index, const uint64_t *codes,
                         const uint8_t *valid, size_t n, uint8_t *hits, unsigned K)
{
    const ProbeParams P = index.probeParams();
    if (index.recordsAre32Bit())
        probeBatched<uint32_t>(codes, valid, n, hits, index.getLut().data(),
                               index.getSuffix32().data(), index.getSignatureMap().data(),
                               index.getNorm().data(), P, K);
    else
        probeBatched<uint64_t>(codes, valid, n, hits, index.getLut().data(),
                               index.getSuffix64().data(), index.getSignatureMap().data(),
                               index.getNorm().data(), P, K);
}

} // namespace fastibs
