#pragma once

// GPU hash-set probe backend (Phase 3 ablation, GPU-HASH): an open-
// addressing table of canonical k-mer codes with double hashing, written
// once as host/device-shared code. The host side is the reference the
// tests validate against KmcRaIndex; the device kernels call the same
// functions. The table stores full 8-byte codes (vs the ported index's
// 2(k-p)/8-byte suffixes), trading footprint for O(1) expected probes —
// exactly the trade the paper's ablation quantifies.
//
// EMPTY (all ones) marks free slots: a canonical code can never be all
// ones, because canonical = min(code, revcomp(code)) and the reverse
// complement of the all-T k-mer is the all-A k-mer (0).
//
// Split across devices (Phase 5), the table becomes one shard per device:
// shard d holds exactly the keys with hashShardOf(key, n) == d and answers
// every other key as a miss, so the OR over the shards is the whole set.

#include <cstdint>

#include "ProbeCore.hpp" // FASTIBS_HD, fastibs::ldg

namespace fastibs
{

constexpr uint64_t HASH_EMPTY = ~0ull;

// splitmix64: cheap, well-distributed 64-bit mixer.
FASTIBS_HD inline uint64_t hashMix(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Capacity is prime, so the double-hash step (1 <= step < capacity) is
// coprime with it and the probe sequence visits every slot. A prime (vs
// power-of-two) capacity also makes the footprint track the requested
// load factor exactly — with rounding to powers of two, a whole range of
// load factors would collapse onto the same table size and the paper's
// footprint-vs-load-factor ablation would be degenerate. The modulo costs
// a few ALU ops per probe step, noise next to the step's DRAM load.
struct HashParams
{
    uint64_t capacity = 0; // prime
    // Shard of the key space this table holds: keys with
    // hashShardOf(key, nShards) == shard. nShards == 1: every key.
    uint32_t shard = 0;
    uint32_t nShards = 1;
};

// Owner shard of a key when the set is split n ways: a salted mix,
// independent of the slot and step hashes, folded to [0, n) by a
// multiply-high (no modulo).
FASTIBS_HD inline uint32_t hashShardOf(uint64_t code, uint32_t nShards)
{
    const uint64_t h = hashMix(code ^ 0x5851F42D4C957F2Dull) >> 32;
    return (uint32_t)((h * nShards) >> 32);
}

FASTIBS_HD inline uint64_t hashSlot0(uint64_t code, const HashParams &P)
{
    return hashMix(code) % P.capacity;
}

FASTIBS_HD inline uint64_t hashStep(uint64_t code, const HashParams &P)
{
    return 1 + hashMix(code ^ 0xA5A5A5A5A5A5A5A5ull) % (P.capacity - 1);
}

FASTIBS_HD inline bool hashContains(uint64_t code, const uint64_t *table,
                                    const HashParams &P)
{
    if (P.nShards > 1 && hashShardOf(code, P.nShards) != P.shard)
        return false; // another device's shard holds this key
    uint64_t slot = hashSlot0(code, P);
    const uint64_t step = hashStep(code, P);
    while (true)
    {
        const uint64_t v = ldg(table + slot);
        if (v == HASH_EMPTY)
            return false; // checked first: the sentinel itself never "hits"
        if (v == code)
            return true;
        slot += step;
        if (slot >= P.capacity)
            slot -= P.capacity;
    }
}

// Host-side insert (device build uses an atomicCAS twin in the .cu).
// Returns false if the key was already present. Callers of a sharded table
// insert only the keys hashShardOf assigns to it.
inline bool hashInsertHost(uint64_t code, uint64_t *table, const HashParams &P)
{
    uint64_t slot = hashSlot0(code, P);
    const uint64_t step = hashStep(code, P);
    while (true)
    {
        if (table[slot] == code)
            return false;
        if (table[slot] == HASH_EMPTY)
        {
            table[slot] = code;
            return true;
        }
        slot += step;
        if (slot >= P.capacity)
            slot -= P.capacity;
    }
}

// Smallest prime capacity holding n keys at most at loadFactor. Callers
// must pass 0 < loadFactor < 1 (DeviceHashIndex validates).
inline uint64_t hashCapacityFor(uint64_t n, double loadFactor)
{
    uint64_t need = (uint64_t)(n / loadFactor) + 1;
    if (need < 7)
        need = 7;
    auto isPrime = [](uint64_t v)
    {
        if (v % 2 == 0)
            return v == 2;
        for (uint64_t d = 3; d * d <= v; d += 2)
            if (v % d == 0)
                return false;
        return true;
    };
    while (!isPrime(need))
        ++need;
    return need;
}

} // namespace fastibs
