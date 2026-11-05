#pragma once

// The KMC membership probe (Algorithm 1 plus the KMC2 minimizer-signature
// bin selection), written once and compiled for both host and device. The
// host side is the validated reference (tests/phase1_test.cpp checks it
// bit-exactly against CKMCFile); every CUDA kernel calls the same template,
// so the search logic cannot drift between the oracle and the GPU.
//
// The structure it searches is either the whole index or one SHARD of it
// (Phase 5, the index split across devices): a shard holds, for every
// signature bin, the records whose p-symbol prefix lies in a range
// [prefixLo, prefixHi), and answers every other query as a miss. The
// shards' ranges partition the prefix space, so exactly one shard holds any
// k-mer and the OR of the shards' answers is the whole index's answer.

#include <cstdint>

#if defined(__CUDACC__)
#define FASTIBS_HD __host__ __device__
#else
#define FASTIBS_HD
#endif

namespace fastibs
{

// Read-only load, routed through the texture path on the device.
template <typename T>
FASTIBS_HD inline T ldg(const T *p)
{
#if defined(__CUDA_ARCH__)
    return __ldg(p);
#else
    return *p;
#endif
}

// Query-path constants of one probe structure, shared by every call site.
struct ProbeParams
{
    uint32_t k;
    uint32_t p;
    uint32_t sigLen; // 0 => KMC1: single LUT, no signature bins
    uint64_t suffixMask;
    uint64_t totalKmers;    // records in THIS structure (a shard's share when split)
    uint64_t singleLutSize; // LUT entries per signature bin: 4^p on disk and for
                            // the whole index; prefixHi - prefixLo + 1 for a
                            // partial shard (KmcRaIndex::shardLayout)
    // Prefix range [prefixLo, prefixHi) this structure holds. The whole
    // index is [0, 4^p), which the defaults cover.
    uint32_t prefixLo = 0;
    uint32_t prefixHi = 0xffffffffu;
};

// Minimizer signature of a canonical k-mer given as a 2-bit packed code
// (A=0..T=3, most significant symbol first). Mirrors
// CKmerAPI::get_signature / CMmer exactly.
FASTIBS_HD inline uint32_t signatureOf(uint64_t code, uint32_t k, uint32_t sigLen,
                                       const uint32_t *norm)
{
    const uint32_t mmerMask = (1u << (2 * sigLen)) - 1;
    uint32_t str = 0;
    uint32_t best = 0xffffffffu;
    for (uint32_t i = 0; i < k; ++i)
    {
        const uint32_t symb = (uint32_t)(code >> (2 * (k - 1 - i))) & 3;
        str = ((str << 2) | symb) & mmerMask;
        if (i >= sigLen - 1)
        {
            const uint32_t v = ldg(norm + str);
            if (v < best)
                best = v;
        }
    }
    return best;
}

// Binary search of one (bin, prefix) bucket: the tail of CKMCFile::CheckKmer
// once the LUT base of the query's signature bin is known. `prefix` is the
// query's p-symbol prefix (already checked to lie in this structure's
// range) and lutBase the bin's first LUT entry.
template <typename RecT>
FASTIBS_HD inline bool probeBucket(uint64_t code, uint64_t prefix, uint64_t lutBase,
                                   const uint64_t *lut, const RecT *suffixes,
                                   const ProbeParams &P)
{
    const uint64_t entry = lutBase + (prefix - P.prefixLo);
    long long lo = (long long)ldg(lut + entry);
    long long hi = (long long)ldg(lut + entry + 1) - 1;
    if (lo >= (long long)P.totalKmers)
        return false;
    if (hi > (long long)P.totalKmers - 1)
        hi = (long long)P.totalKmers - 1;

    const RecT target = (RecT)(code & P.suffixMask);
    while (lo <= hi)
    {
        const long long mid = (lo + hi) >> 1;
        const RecT s = ldg(suffixes + mid);
        if (s == target)
            return true;
        if (s < target)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return false;
}

// Membership probe mirroring CKMCFile::CheckKmer: signature -> bin ->
// two LUT loads -> binary search over the bucket's counter-stripped
// suffix records (one integer compare per step).
template <typename RecT>
FASTIBS_HD inline bool probeOne(uint64_t code, const uint64_t *lut,
                                const RecT *suffixes, const uint32_t *sigMap,
                                const uint32_t *norm, const ProbeParams &P)
{
    const uint64_t prefix = code >> (2 * (P.k - P.p));
    // Another device's shard holds this prefix range. Decided before the
    // signature scan, so a query that is not ours costs only the encode.
    if (prefix < P.prefixLo || prefix >= P.prefixHi)
        return false;

    uint64_t lutBase = 0;
    if (P.sigLen)
        lutBase = (uint64_t)ldg(sigMap + signatureOf(code, P.k, P.sigLen, norm)) *
                  P.singleLutSize;
    return probeBucket<RecT>(code, prefix, lutBase, lut, suffixes, P);
}

// The same probe with the signature already computed (sig =
// signatureOf(code, P.k, P.sigLen, norm); ignored for KMC1). The minimizer
// signature of a k-mer depends on k and the signature length only, not on
// the database, so a query asked of several databases that share them
// (panel mode, docs/MULTIDB_SCAN.md) computes it once and pays per
// database only the signature-map load, the two LUT loads and the bucket
// search. probeOne is exactly signatureOf followed by this function.
template <typename RecT>
FASTIBS_HD inline bool probeOneWithSig(uint64_t code, uint32_t sig, const uint64_t *lut,
                                       const RecT *suffixes, const uint32_t *sigMap,
                                       const ProbeParams &P)
{
    const uint64_t prefix = code >> (2 * (P.k - P.p));
    if (prefix < P.prefixLo || prefix >= P.prefixHi)
        return false;
    uint64_t lutBase = 0;
    if (P.sigLen)
        lutBase = (uint64_t)ldg(sigMap + sig) * P.singleLutSize;
    return probeBucket<RecT>(code, prefix, lutBase, lut, suffixes, P);
}

} // namespace fastibs
