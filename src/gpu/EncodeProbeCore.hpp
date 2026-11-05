#pragma once

// Core logic of the Phase 2 fused encode-and-probe pipeline, shared
// between host and device:
//
//  - the batch layout (window descriptors over concatenated, word-aligned
//    2-bit packed window sequences),
//  - host-side packing (2-bit codes + non-ACGT exception bitmask, with
//    case folding in the table — the "ship sequence, not k-mers" format:
//    W/4 bytes of sequence per window instead of 8(W-k+1) bytes of
//    expanded queries),
//  - the per-output-word fused kernel body (rolling 2-bit encode +
//    canonicalization + immediate probe, 32 positions per thread, the
//    thread exclusively owning one 32-bit word of the hit mask and one of
//    the validity mask — no atomics), and
//  - the host statistics pass over the returned bit masks.
//
// The kernel body is a template over the probe functor: the CUDA kernel
// instantiates it with the device index, the tests instantiate it with the
// host KmcRaIndex (or any predicate), so the logic validated on the CPU is
// bit-for-bit the logic the GPU runs.
//
// Encoding conventions match KmerCodec exactly: A=0 C=1 G=2 T=3, forward
// code rolls in at the low end (first base in the high bits), reverse
// complement maintained from the high end, canonical = numeric min,
// positions whose k-mer window covers a non-ACGT byte are invalid.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "ProbeCore.hpp" // FASTIBS_HD, fastibs::ldg

namespace fastibs
{

// Placement of one window inside a batch. Sequence words are 32 bases
// per uint64 (2 bits each, base b at bits 2*(b%32)); exception-mask words
// are 64 bases per uint64 (bit b%64 set when base b is not ACGT); output
// words are 32 positions per uint32 (bit p%32).
struct WindowDesc
{
    uint64_t seqWordOff; // first sequence word of this window
    uint64_t invWordOff; // first exception-mask word
    uint64_t outWordOff; // first hit/valid output word
    uint32_t nBases;     // window length in bases
    uint32_t nPos;       // output positions: nBases - k + 1 (0 if shorter)
};

struct BatchLayout
{
    std::vector<WindowDesc> descs;
    uint64_t seqWords = 0; // total uint64 sequence words
    uint64_t invWords = 0; // total uint64 exception-mask words
    uint64_t outWords = 0; // total uint32 output words (per mask)

    void clear()
    {
        descs.clear();
        seqWords = invWords = outWords = 0;
    }

    // Appends a window of nBases; returns its descriptor index.
    size_t add(uint32_t nBases, uint32_t k)
    {
        WindowDesc d;
        d.seqWordOff = seqWords;
        d.invWordOff = invWords;
        d.outWordOff = outWords;
        d.nBases = nBases;
        d.nPos = nBases >= k ? nBases - k + 1 : 0;
        descs.push_back(d);
        seqWords += (nBases + 31) / 32;
        invWords += (nBases + 63) / 64;
        outWords += (d.nPos + 31) / 32;
        return descs.size() - 1;
    }
};

// Slot capacities, in units of the batch layout.
struct BatchCaps
{
    size_t seqWords;
    size_t invWords;
    size_t outWords;
    size_t windows;
};

// One batch of windows: indices into the caller's window list plus the
// packed layout.
struct WindowBatch
{
    std::vector<size_t> winIdx;
    BatchLayout layout;
};

// Splits windows into batches that each fit `caps`, invoking flush(batch)
// for every completed (never empty) batch. Windows shorter than k are
// skipped (they produce no queries). Throws if a single window cannot fit
// an empty batch — callers must size the capacities to the largest window.
template <typename LenFn, typename FlushFn>
inline void buildBatches(size_t nWindows, uint32_t k, const BatchCaps &caps,
                         LenFn &&lenOf, FlushFn &&flush)
{
    WindowBatch building;
    for (size_t i = 0; i < nWindows; ++i)
    {
        const size_t nBases = lenOf(i);
        if (nBases < k)
            continue;
        const uint64_t needSeq = (nBases + 31) / 32;
        const uint64_t needInv = (nBases + 63) / 64;
        const uint64_t needOut = (nBases - k + 1 + 31) / 32;
        if (needSeq > caps.seqWords || needInv > caps.invWords ||
            needOut > caps.outWords || caps.windows < 1)
            throw std::runtime_error(
                "buildBatches: a single window exceeds the batch capacity");
        const bool fits = building.winIdx.size() + 1 <= caps.windows &&
                          building.layout.seqWords + needSeq <= caps.seqWords &&
                          building.layout.invWords + needInv <= caps.invWords &&
                          building.layout.outWords + needOut <= caps.outWords;
        if (!fits)
        {
            flush(std::move(building));
            building = WindowBatch{};
        }
        building.layout.add((uint32_t)nBases, k);
        building.winIdx.push_back(i);
    }
    if (!building.winIdx.empty())
        flush(std::move(building));
}

// Base-to-code table with case folding; 4 marks a non-ACGT byte.
// (Same mapping as KmerCodec::baseCode, duplicated here so this header
// stays free of host-only includes for device compilation.)
inline const uint8_t *packTable()
{
    static const uint8_t *t = []
    {
        static uint8_t tab[256];
        memset(tab, 4, sizeof(tab));
        tab[(uint8_t)'A'] = tab[(uint8_t)'a'] = 0;
        tab[(uint8_t)'C'] = tab[(uint8_t)'c'] = 1;
        tab[(uint8_t)'G'] = tab[(uint8_t)'g'] = 2;
        tab[(uint8_t)'T'] = tab[(uint8_t)'t'] = 3;
        return tab;
    }();
    return t;
}

// Packs one window's sequence into its (word-aligned) region of the batch
// buffers. seqWords/invWords point at the window's own first words; the
// final partial words are zero-padded. Host-only.
inline void packWindow(std::string_view seq, uint64_t *seqWords, uint64_t *invWords)
{
    const uint8_t *tab = packTable();
    const size_t n = seq.size();
    uint64_t sw = 0, iw = 0;
    for (size_t b = 0; b < n; ++b)
    {
        uint8_t c = tab[(uint8_t)seq[b]];
        if (c > 3)
        {
            iw |= 1ull << (b & 63);
            c = 0;
        }
        sw |= (uint64_t)c << (2 * (b & 31));
        if ((b & 31) == 31)
        {
            seqWords[b / 32] = sw;
            sw = 0;
        }
        if ((b & 63) == 63)
        {
            invWords[b / 64] = iw;
            iw = 0;
        }
    }
    if (n & 31)
        seqWords[n / 32] = sw;
    if (n & 63)
        invWords[n / 64] = iw;
}

// Locates the window containing flat output word `wordIdx`: the last
// descriptor with outWordOff <= wordIdx. Descriptors are ordered by
// outWordOff; zero-position windows never own an output word but may
// share outWordOff with their successor, which the low-side bound
// handles (the found window always has nPos > 0 for an in-range word).
FASTIBS_HD inline uint32_t findWindow(uint64_t wordIdx, const WindowDesc *descs,
                                      uint32_t nWindows)
{
    uint32_t lo = 0, hi = nWindows - 1;
    while (lo < hi)
    {
        const uint32_t mid = (lo + hi + 1) >> 1;
        if (ldg(&descs[mid].outWordOff) <= wordIdx)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// The K1 body for one output word: rolling 2-bit encode + canonicalization
// over up to 32 consecutive positions of one window, handing every emitted
// position to `emit(bit, code, isValid)` (code is unspecified when
// invalid). Returns the validity word. Padding bits beyond the window's
// last position are never emitted.
template <typename EmitFn>
FASTIBS_HD inline uint32_t encodeWord(uint64_t wordIdx, const WindowDesc *descs,
                                      uint32_t nWindows,
                                      const uint64_t *seqWords,
                                      const uint64_t *invWords,
                                      uint32_t k, EmitFn &&emit)
{
    const uint32_t w = findWindow(wordIdx, descs, nWindows);
    const uint64_t seqOff = descs[w].seqWordOff;
    const uint64_t invOff = descs[w].invWordOff;
    const uint32_t nPos = descs[w].nPos;
    const uint32_t p0 = (uint32_t)((wordIdx - descs[w].outWordOff) * 32);
    const uint32_t emitCount = nPos - p0 < 32 ? nPos - p0 : 32;

    const uint64_t kmask = k == 32 ? ~0ull : (1ull << (2 * k)) - 1;
    const uint32_t rcShift = 2 * (k - 1);

    uint64_t fwd = 0, rc = 0;
    uint32_t validBases = 0;
    uint32_t valids = 0;

    // Bases [p0, p0 + emitCount + k - 2] of the window, read sequentially
    // with a manually-advanced word cursor (neighboring threads read
    // overlapping, adjacent words — L2-friendly).
    uint64_t sw = ldg(seqWords + seqOff + p0 / 32);
    uint64_t iw = ldg(invWords + invOff + p0 / 64);
    for (uint32_t j = 0; j < emitCount + k - 1; ++j)
    {
        const uint32_t b = p0 + j;
        if (j && (b & 31) == 0)
            sw = ldg(seqWords + seqOff + b / 32);
        if (j && (b & 63) == 0)
            iw = ldg(invWords + invOff + b / 64);

        const uint32_t c = (uint32_t)(sw >> (2 * (b & 31))) & 3;
        if ((iw >> (b & 63)) & 1)
            validBases = 0;
        else
            ++validBases;
        fwd = ((fwd << 2) | c) & kmask;
        rc = (rc >> 2) | ((uint64_t)(3 - c) << rcShift);

        if (j >= k - 1)
        {
            const uint32_t bit = j - (k - 1);
            const bool ok = validBases >= k;
            if (ok)
                valids |= 1u << bit;
            emit(bit, fwd < rc ? fwd : rc, ok);
        }
    }
    return valids;
}

// The fused K1+K2 body: probes each valid position immediately (the
// canonical code lives only in registers). Writes one hit word and one
// validity word; padding bits beyond the window's last position stay 0 in
// both masks.
template <typename ProbeFn>
FASTIBS_HD inline void encodeProbeWord(uint64_t wordIdx, const WindowDesc *descs,
                                       uint32_t nWindows,
                                       const uint64_t *seqWords,
                                       const uint64_t *invWords,
                                       uint32_t k, ProbeFn &&probe,
                                       uint32_t &hitWord, uint32_t &validWord)
{
    uint32_t hits = 0;
    validWord = encodeWord(wordIdx, descs, nWindows, seqWords, invWords, k,
                           [&](uint32_t bit, uint64_t code, bool ok)
                           {
                               if (ok && probe(code))
                                   hits |= 1u << bit;
                           });
    hitWord = hits;
}

// Databases a panel-mode batch is probed against at once (the kernels take
// that many probe views by value; docs/MULTIDB_SCAN.md section 3).
constexpr uint32_t MAX_DBS = 16;

// The fused body for N databases (panel mode): ONE encode per position,
// N membership probes. `probe` provides count() (n <= MAX_DBS),
// prepare(code) -> the database-independent part of the search (the
// minimizer signature when the databases share k and the signature length,
// else 0) computed once per code, and operator()(d, code, prep) -> the
// membership of code in database d. Hit bits accumulate in one word per
// database (a per-thread array; local memory on the device) and are
// written out once at the end to hitWords[d * hitStride + wordIdx]; the
// validity word is computed once, as it depends on the reference only.
// With n == 1 the hit word is bit-for-bit encodeProbeWord's.
template <typename MultiProbeFn>
FASTIBS_HD inline void encodeProbeWordMulti(uint64_t wordIdx, const WindowDesc *descs,
                                            uint32_t nWindows,
                                            const uint64_t *seqWords,
                                            const uint64_t *invWords,
                                            uint32_t k, MultiProbeFn &&probe,
                                            uint32_t *hitWords, size_t hitStride,
                                            uint32_t &validWord)
{
    const uint32_t n = probe.count();
    uint32_t hits[MAX_DBS];
    for (uint32_t d = 0; d < MAX_DBS; ++d)
        hits[d] = 0;
    validWord = encodeWord(wordIdx, descs, nWindows, seqWords, invWords, k,
                           [&](uint32_t bit, uint64_t code, bool ok)
                           {
                               if (!ok)
                                   return;
                               const uint32_t prep = probe.prepare(code);
                               for (uint32_t d = 0; d < n; ++d)
                                   if (probe(d, code, prep))
                                       hits[d] |= 1u << bit;
                           });
    for (uint32_t d = 0; d < n; ++d)
        hitWords[(size_t)d * hitStride + wordIdx] = hits[d];
}

// Host statistics pass over one window's returned bit masks: iterates the
// window's positions in order, skipping invalid ones, and applies the
// same gap run-length accumulation as the CPU probe loop. Invalid
// positions do not appear in the valid-k-mer stream, so a gap continues
// across them — identical semantics to the codec's compacted output.
// Accum must provide: onTotalKmer(), onHit(gapSize), gap closure via
// finish(gapSize) — see the caller; kept as a callback-free concrete
// summary instead:
struct BitStats
{
    uint64_t totalKmers = 0;
    uint64_t observedKmers = 0;
    // gap runs (of misses, in valid-position space), reported via
    // gaps(): call finalize() after the last position.
    template <typename OnGap>
    static void run(const uint32_t *hitWords, const uint32_t *validWords,
                    uint64_t firstWord, uint32_t nPos, BitStats &out, OnGap &&onGap)
    {
        uint64_t gap = 0;
        for (uint32_t p = 0; p < nPos; ++p)
        {
            const uint32_t word = (uint32_t)(p / 32), bit = p & 31;
            if (!((validWords[firstWord + word] >> bit) & 1))
                continue;
            out.totalKmers++;
            if ((hitWords[firstWord + word] >> bit) & 1)
            {
                out.observedKmers++;
                if (gap > 0)
                {
                    onGap(gap);
                    gap = 0;
                }
            }
            else
                gap++;
        }
        if (gap > 0)
            onGap(gap);
    }
};

} // namespace fastibs
