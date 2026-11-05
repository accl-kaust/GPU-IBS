#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <array>

// Rolling 2-bit codec: produces, per reference position, the canonical
// 64-bit packed k-mer in O(1) — the forward code by shift-and-mask, the
// reverse complement maintained incrementally from the high end, and
// canonical = min of the two. With the encoding A=0, C=1, G=2, T=3,
// lexicographic string order equals numeric order of the packed integer,
// so this reproduces the canonical form of the string implementation
// exactly. Upper-casing is folded into the base-to-code table.
// Supports k <= 32.
class KmerCodec
{
public:
    static constexpr uint8_t INVALID_BASE = 4;

    explicit KmerCodec(uint32_t kmerSize)
        : k(kmerSize),
          mask(kmerSize == 32 ? ~0ull : (1ull << (2 * kmerSize)) - 1),
          rcShift(2 * (kmerSize - 1))
    {
    }

    static uint8_t baseCode(char base)
    {
        return table()[static_cast<uint8_t>(base)];
    }

    // Canonical codes of all valid positions, in order; positions whose
    // k-mer window covers a non-ACGT byte are skipped (same compaction as
    // the legacy string-based getCanonicalKmers).
    void getCanonicalCodes(std::string_view sequence, std::vector<uint64_t> &codes) const
    {
        codes.clear();
        if (sequence.size() < k)
            return;
        codes.reserve(sequence.size() - k + 1);
        uint64_t fwd = 0, rc = 0;
        uint32_t validBases = 0;
        for (size_t j = 0; j < sequence.size(); ++j)
        {
            uint8_t c = table()[static_cast<uint8_t>(sequence[j])];
            if (c > 3)
            {
                validBases = 0;
                c = 0; // garbage bits flush out after k valid bases
            }
            else
                ++validBases;
            fwd = ((fwd << 2) | c) & mask;
            rc = (rc >> 2) | (static_cast<uint64_t>(3 - c) << rcShift);
            if (validBases >= k)
                codes.push_back(fwd < rc ? fwd : rc);
        }
    }

    // Per-position variant: codes[i] holds the canonical code of the k-mer
    // starting at position i and valid[i] whether its window is ACGT-only
    // (codes[i] is unspecified when valid[i] == 0). Both are sized
    // N - k + 1; empty when the sequence is shorter than k.
    void encodePositions(std::string_view sequence,
                         std::vector<uint64_t> &codes,
                         std::vector<uint8_t> &valid) const
    {
        codes.clear();
        valid.clear();
        if (sequence.size() < k)
            return;
        const size_t n = sequence.size() - k + 1;
        codes.resize(n);
        valid.resize(n);
        uint64_t fwd = 0, rc = 0;
        uint32_t validBases = 0;
        for (size_t j = 0; j < sequence.size(); ++j)
        {
            uint8_t c = table()[static_cast<uint8_t>(sequence[j])];
            if (c > 3)
            {
                validBases = 0;
                c = 0;
            }
            else
                ++validBases;
            fwd = ((fwd << 2) | c) & mask;
            rc = (rc >> 2) | (static_cast<uint64_t>(3 - c) << rcShift);
            if (j >= k - 1)
            {
                const size_t i = j - (k - 1);
                const bool ok = validBases >= k;
                valid[i] = ok;
                codes[i] = ok ? (fwd < rc ? fwd : rc) : 0;
            }
        }
    }

    uint32_t kmerSize() const { return k; }

private:
    static const std::array<uint8_t, 256> &table()
    {
        static const std::array<uint8_t, 256> t = []
        {
            std::array<uint8_t, 256> t{};
            t.fill(INVALID_BASE);
            t['A'] = t['a'] = 0;
            t['C'] = t['c'] = 1;
            t['G'] = t['g'] = 2;
            t['T'] = t['t'] = 3;
            return t;
        }();
        return t;
    }

    uint32_t k;
    uint64_t mask;
    uint32_t rcShift;
};
