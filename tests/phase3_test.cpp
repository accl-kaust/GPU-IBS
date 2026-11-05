// Phase 3 host-side regression test: validates the ablation building
// blocks whose logic is shared host/device —
//   (1) KmcRaIndex::forEachKmer reconstructs exactly the database's k-mer
//       set (checked against a CKMCFile listing pass), and the parallel
//       enumeration of KDBIntersect (ParallelEnum.hpp) delivers exactly
//       that set to its probe batches for every producer count, sink
//       count and batch size tried, routed to the right sink, with a
//       failing probe or owner stopping every thread instead of hanging,
//       an idle sink finishing cleanly, and slow consumers (back-pressure
//       on the bounded queue) still delivering the set — run it on a KMC2
//       database and on a KMC1-format one (kmc_tools output) to cover
//       both LUT layouts,
//   (2) the hash-set backend (HashCore, the same hashContains the GPU
//       kernels call) agrees with the ported-index probe on present and
//       random k-mers across load factors,
//   (3) the sorted path (encode -> value sort with position payload ->
//       probe -> scatter) produces bit-identical hit/valid words to the
//       fused path.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a tests/phase3_test.cpp KMC/kmc_api/kmc_file.cpp \
//       KMC/kmc_api/kmer_api.cpp KMC/kmc_api/mmer.cpp -lz -lpthread -o phase3_test
// Run:
//   ./phase3_test <kmc_db_path_prefix>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../src/Utils.hpp"
#include "../src/gpu/EncodeProbeCore.hpp"
#include "../src/gpu/HashCore.hpp"
#include "../src/gpu/KmcRaIndex.hpp"
#include "../src/gpu/ParallelEnum.hpp"
#include "../KMC/kmc_api/kmc_file.h"

static int failures = 0;
#define CHECK(cond, msg)                                   \
    do                                                     \
    {                                                      \
        if (!(cond))                                       \
        {                                                  \
            printf("FAIL: %s (line %d)\n", msg, __LINE__); \
            failures++;                                    \
        }                                                  \
    } while (0)

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: %s <kmc_db_path_prefix>\n", argv[0]);
        return 1;
    }
    const std::string db = argv[1];
    KmcRaIndex index(db);
    const uint32_t k = index.getParams().kmerLength;
    const uint64_t total = index.getParams().totalKmers;
    printf("db: k=%u total=%llu\n", k, (unsigned long long)total);

    // --- 1. enumeration == listing -------------------------------------
    std::vector<uint64_t> enumerated;
    enumerated.reserve(total);
    index.forEachKmer([&](uint64_t code) { enumerated.push_back(code); });
    CHECK(enumerated.size() == total, "enumeration count mismatch");

    std::vector<uint64_t> listed;
    listed.reserve(total);
    {
        CKMCFile lister;
        if (!lister.OpenForListing(db))
        {
            printf("cannot open %s for listing\n", db.c_str());
            return 1;
        }
        CKmerAPI kmer(k);
        uint32 count;
        std::vector<uint64> words;
        while (lister.ReadNextKmer(kmer, count))
        {
            kmer.to_long(words);
            listed.push_back(words[0]);
        }
    }
    std::sort(enumerated.begin(), enumerated.end());
    std::sort(listed.begin(), listed.end());
    CHECK(enumerated == listed, "enumerated k-mer set != listed k-mer set");
    printf("enumeration matches listing: %zu k-mers\n", enumerated.size());

    // --- 1b. parallel enumeration == sequential -------------------------
    // KDBIntersect's producer/consumer driver: the probe batches must
    // carry exactly the k-mer set (every k-mer once), each at the sink its
    // owner function names, for odd batch sizes (a bucket larger than a
    // unit, a unit of many buckets) and producer/sink counts.
    for (unsigned producers : {1u, 2u, 5u})
        for (int sinks : {1, 3})
            for (size_t batch : {(size_t)7, (size_t)1000, (size_t)1 << 22})
            {
                std::mutex m;
                std::vector<uint64_t> got;
                uint64_t misrouted = 0;
                fastibs::EnumCountStats st;
                const uint64_t count = fastibs::enumerateAndCount(
                    index, producers, sinks, batch, 2,
                    [&](uint64_t code) { return (int)(code % (uint64_t)sinks); },
                    [&](int sink, const std::vector<uint64_t> &keys)
                    {
                        uint64_t wrong = 0;
                        for (uint64_t c : keys)
                            wrong += (int)(c % (uint64_t)sinks) != sink;
                        std::lock_guard<std::mutex> lk(m);
                        misrouted += wrong;
                        got.insert(got.end(), keys.begin(), keys.end());
                        return (uint64_t)keys.size();
                    },
                    &st);
                CHECK(count == total, "parallel enumeration: count != total");
                CHECK(misrouted == 0, "parallel enumeration: key at the wrong sink");
                std::sort(got.begin(), got.end());
                CHECK(got == enumerated, "parallel enumeration != sequential set");
                CHECK(st.keys == total && st.producers == producers &&
                          st.consumers == 2u * (unsigned)sinks && st.batchKeys == batch &&
                          st.batches >= (total + batch - 1) / batch,
                      "parallel enumeration: stats disagree");
            }
    printf("parallel enumeration matches for 3 producer x 2 sink x 3 batch settings\n");

    // A failing probe must stop producers and consumers and surface, not
    // hang on a full or empty queue.
    {
        bool thrown = false;
        try
        {
            fastibs::enumerateAndCount(
                index, 4, 2, 1000, 1, [](uint64_t code) { return (int)(code & 1); },
                [](int, const std::vector<uint64_t> &) -> uint64_t
                { throw std::runtime_error("probe failed"); });
        }
        catch (const std::runtime_error &e)
        {
            thrown = std::string(e.what()) == "probe failed";
        }
        CHECK(thrown, "parallel enumeration: probe failure not propagated");
        printf("parallel enumeration: a failing probe stops the run\n");
    }

    // A failure on the producer side (the owner function) must surface the
    // same way, with the consumers released from their empty queues.
    {
        bool thrown = false;
        try
        {
            fastibs::enumerateAndCount(
                index, 3, 2, 1000, 2,
                [](uint64_t) -> int { throw std::runtime_error("owner failed"); },
                [](int, const std::vector<uint64_t> &keys) { return (uint64_t)keys.size(); });
        }
        catch (const std::runtime_error &e)
        {
            thrown = std::string(e.what()) == "owner failed";
        }
        CHECK(thrown, "parallel enumeration: owner failure not propagated");
        printf("parallel enumeration: a failing owner stops the run\n");
    }

    // A sink that never receives a batch (the owner names only sinks 0 and
    // 1 of 3): its consumers must simply finish, and the count stays whole.
    {
        std::atomic<uint64_t> seen{0};
        std::atomic<int> sink2Batches{0};
        fastibs::EnumCountStats st;
        const uint64_t count = fastibs::enumerateAndCount(
            index, 3, 3, 1000, 1, [](uint64_t code) { return (int)(code & 1); },
            [&](int sink, const std::vector<uint64_t> &keys)
            {
                if (sink == 2)
                    ++sink2Batches;
                seen += keys.size();
                return (uint64_t)keys.size();
            },
            &st);
        CHECK(count == total && seen == total, "idle sink: count != total");
        CHECK(sink2Batches == 0, "idle sink received a batch");
        printf("parallel enumeration: an idle sink finishes cleanly\n");
    }

    // Slow consumers: with small batches and a probe that sleeps, the
    // producers fill the bounded queue and wait on it (back-pressure); the
    // run must still deliver every k-mer exactly once.
    {
        std::mutex m;
        std::vector<uint64_t> got;
        const uint64_t count = fastibs::enumerateAndCount(
            index, 4, 1, 50000, 2, [](uint64_t) { return 0; },
            [&](int, const std::vector<uint64_t> &keys)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                std::lock_guard<std::mutex> lk(m);
                got.insert(got.end(), keys.begin(), keys.end());
                return (uint64_t)keys.size();
            });
        std::sort(got.begin(), got.end());
        CHECK(count == total && got == enumerated,
              "slow consumers: delivered set != sequential set");
        printf("parallel enumeration: slow consumers (back-pressure) deliver the set\n");
    }

    // no canonical code may collide with the hash-empty sentinel
    CHECK(std::find(enumerated.begin(), enumerated.end(), fastibs::HASH_EMPTY) ==
              enumerated.end(),
          "sentinel collides with a canonical code");

    // --- 2. hash backend equivalence ------------------------------------
    std::mt19937_64 rng(20260901);
    const uint64_t kmerMask = k == 32 ? ~0ull : (1ull << (2 * k)) - 1;
    for (double lf : {0.4, 0.6, 0.9})
    {
        const uint64_t cap = fastibs::hashCapacityFor(total, lf);
        // prime capacity: the footprint must actually track the requested
        // load factor (the previous power-of-two rounding collapsed whole
        // ranges of load factors onto one table size)
        const double actualLf = (double)total / cap;
        CHECK(actualLf <= lf && actualLf > lf * 0.98,
              "capacity does not track the requested load factor");
        fastibs::HashParams HP{cap};
        std::vector<uint64_t> table(cap, fastibs::HASH_EMPTY);
        uint64_t inserted = 0;
        index.forEachKmer([&](uint64_t code)
                          { inserted += fastibs::hashInsertHost(code, table.data(), HP); });
        CHECK(inserted == total, "hash insert count mismatch");

        uint64_t presentOk = 0;
        for (uint64_t code : enumerated)
            presentOk += fastibs::hashContains(code, table.data(), HP);
        CHECK(presentOk == total, "present k-mer missing from hash");

        // the sentinel itself must never report a hit
        CHECK(!fastibs::hashContains(fastibs::HASH_EMPTY, table.data(), HP),
              "sentinel query reported a hit");

        uint64_t agree = 0, hits = 0;
        const int N = 500000;
        for (int i = 0; i < N; i++)
        {
            const uint64_t code = rng() & kmerMask;
            const bool a = index.probe(code);
            const bool b = fastibs::hashContains(code, table.data(), HP);
            agree += a == b;
            hits += a;
        }
        CHECK(agree == (uint64_t)N, "hash/index disagreement on random k-mers");
        printf("hash lf=%.1f: capacity %llu (actual lf %.3f), all present found, %d/%d random agree (%llu hits)\n",
               lf, (unsigned long long)cap, actualLf, N, N,
               (unsigned long long)hits);
    }

    // --- 3. sorted path == fused path ------------------------------------
    {
        static const char bases[] = "ACGTacgtN";
        std::vector<std::string> ids{"c0", "c1"};
        std::vector<std::string> seqs;
        for (size_t len : {100000ul, 3333ul})
        {
            std::string s(len, 'A');
            for (auto &c : s)
                c = bases[rng() % 9 == 0 ? 8 : rng() % 8]; // ~11% N
            seqs.push_back(std::move(s));
        }
        std::vector<SeqWindow> windows;
        chunkSequences(ids, seqs, 5000, k, windows);

        fastibs::BatchLayout layout;
        for (const auto &w : windows)
            if (w.seq.size() >= k)
                layout.add((uint32_t)w.seq.size(), k);
        std::vector<uint64_t> seqWords(layout.seqWords, 0);
        std::vector<uint64_t> invWords(layout.invWords, 0);
        size_t j = 0;
        for (const auto &w : windows)
            if (w.seq.size() >= k)
            {
                const auto &d = layout.descs[j++];
                fastibs::packWindow(w.seq, seqWords.data() + d.seqWordOff,
                                    invWords.data() + d.invWordOff);
            }

        auto probe = [&](uint64_t code) { return index.probe(code); };
        const uint32_t nW = (uint32_t)layout.descs.size();

        // fused reference
        std::vector<uint32_t> fusedHit(layout.outWords), fusedValid(layout.outWords);
        for (uint64_t w = 0; w < layout.outWords; w++)
            fastibs::encodeProbeWord(w, layout.descs.data(), nW, seqWords.data(),
                                     invWords.data(), k, probe, fusedHit[w],
                                     fusedValid[w]);

        // sorted emulation: collect -> sort by value (position payload) ->
        // probe in sorted order -> scatter
        const uint64_t SENTINEL = ~0ull;
        const size_t P = layout.outWords * 32;
        std::vector<uint64_t> codes(P, SENTINEL);
        std::vector<uint32_t> sortedValid(layout.outWords);
        for (uint64_t w = 0; w < layout.outWords; w++)
            sortedValid[w] = fastibs::encodeWord(
                w, layout.descs.data(), nW, seqWords.data(), invWords.data(), k,
                [&](uint32_t bit, uint64_t code, bool ok)
                { codes[w * 32 + bit] = ok ? code : SENTINEL; });
        std::vector<uint32_t> pos(P);
        std::iota(pos.begin(), pos.end(), 0);
        std::sort(pos.begin(), pos.end(), [&](uint32_t a, uint32_t b)
                  { return codes[a] < codes[b]; });
        std::vector<uint32_t> sortedHit(layout.outWords, 0);
        for (size_t i = 0; i < P; i++)
        {
            const uint64_t code = codes[pos[i]];
            if (code == SENTINEL)
                continue;
            if (probe(code))
                sortedHit[pos[i] >> 5] |= 1u << (pos[i] & 31);
        }

        CHECK(sortedValid == fusedValid, "sorted path valid words differ");
        CHECK(sortedHit == fusedHit, "sorted path hit words differ");
        printf("sorted path: %llu words bit-identical to fused path\n",
               (unsigned long long)layout.outWords);
    }

    if (failures)
    {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL PHASE 3 HOST TESTS PASSED\n");
    return 0;
}
