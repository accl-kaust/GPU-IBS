// Phase 2 host-side regression test: validates the fused encode+probe
// pipeline logic — batch layout, 2-bit packing with exception masks,
// the per-word fused kernel body (encodeProbeWord), the statistics pass
// over bit masks, and the capacity-driven batch splitting (buildBatches)
// — against the Phase 0 codec reference. The CUDA kernel instantiates the
// same encodeProbeWord/probeOne templates this test drives on the host,
// and KmerDatabase's pipeline uses the same buildBatches; what remains
// untested here is only the CUDA stream/copy mechanics and the
// thread-pool interleaving of the live pipeline.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a tests/phase2_test.cpp KMC/kmc_api/kmc_file.cpp \
//       KMC/kmc_api/kmer_api.cpp KMC/kmc_api/mmer.cpp -lz -o phase2_test
// Run:
//   ./phase2_test <kmc_db_path_prefix>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/KmerCodec.hpp"
#include "../src/Utils.hpp"
#include "../src/gpu/EncodeProbeCore.hpp"
#include "../src/gpu/KmcRaIndex.hpp"

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

struct RefStats
{
    uint64_t total = 0, observed = 0;
    std::vector<uint64_t> gaps;
};

// Reference semantics: Phase 0 codec compaction + run-length loop.
template <typename ProbeFn>
RefStats referenceStats(std::string_view seq, uint32_t k, ProbeFn &&probe)
{
    RefStats r;
    std::vector<uint64_t> codes;
    KmerCodec(k).getCanonicalCodes(seq, codes);
    r.total = codes.size();
    uint64_t gap = 0;
    for (uint64_t code : codes)
    {
        if (probe(code))
        {
            r.observed++;
            if (gap)
            {
                r.gaps.push_back(gap);
                gap = 0;
            }
        }
        else
            gap++;
    }
    if (gap)
        r.gaps.push_back(gap);
    return r;
}

// Emulated pipeline: pack all windows into one batch, run the fused kernel
// body per output word exactly as the GPU does, then the stats pass.
template <typename ProbeFn>
std::vector<RefStats> pipelineStats(const std::vector<SeqWindow> &windows,
                                    uint32_t k, ProbeFn &&probe)
{
    fastibs::BatchLayout layout;
    std::vector<size_t> probed; // indices of windows with nPos > 0
    for (size_t i = 0; i < windows.size(); i++)
    {
        if (windows[i].seq.size() < k)
            continue;
        layout.add((uint32_t)windows[i].seq.size(), k);
        probed.push_back(i);
    }

    std::vector<uint64_t> seqWords(layout.seqWords, 0);
    std::vector<uint64_t> invWords(layout.invWords, 0);
    for (size_t j = 0; j < probed.size(); j++)
    {
        const auto &d = layout.descs[j];
        fastibs::packWindow(windows[probed[j]].seq, seqWords.data() + d.seqWordOff,
                            invWords.data() + d.invWordOff);
    }

    std::vector<uint32_t> hitWords(layout.outWords, 0);
    std::vector<uint32_t> validWords(layout.outWords, 0);
    for (uint64_t w = 0; w < layout.outWords; w++)
        fastibs::encodeProbeWord(w, layout.descs.data(),
                                 (uint32_t)layout.descs.size(), seqWords.data(),
                                 invWords.data(), k, probe, hitWords[w],
                                 validWords[w]);

    std::vector<RefStats> out(windows.size());
    for (size_t j = 0; j < probed.size(); j++)
    {
        const auto &d = layout.descs[j];
        fastibs::BitStats bs;
        RefStats &r = out[probed[j]];
        fastibs::BitStats::run(hitWords.data(), validWords.data(), d.outWordOff,
                               d.nPos, bs, [&](uint64_t g)
                               { r.gaps.push_back(g); });
        r.total = bs.totalKmers;
        r.observed = bs.observedKmers;
    }
    return out;
}

static bool same(const RefStats &a, const RefStats &b)
{
    return a.total == b.total && a.observed == b.observed && a.gaps == b.gaps;
}

// Batched emulation: split windows into capacity-bounded batches exactly as
// the live pipeline does (fastibs::buildBatches), process every batch
// independently, and merge — batch boundaries must not affect any result.
template <typename ProbeFn>
std::vector<RefStats> pipelineStatsBatched(const std::vector<SeqWindow> &windows,
                                           uint32_t k, const fastibs::BatchCaps &caps,
                                           ProbeFn &&probe, size_t &nBatches)
{
    std::vector<RefStats> out(windows.size());
    nBatches = 0;
    fastibs::buildBatches(
        windows.size(), k, caps,
        [&](size_t i) { return windows[i].seq.size(); },
        [&](fastibs::WindowBatch &&b)
        {
            nBatches++;
            if (b.winIdx.size() > caps.windows ||
                b.layout.seqWords > caps.seqWords ||
                b.layout.invWords > caps.invWords ||
                b.layout.outWords > caps.outWords)
            {
                printf("FAIL: emitted batch exceeds caps\n");
                failures++;
            }
            std::vector<uint64_t> seqWords(b.layout.seqWords, ~0ull); // dirty
            std::vector<uint64_t> invWords(b.layout.invWords, ~0ull); // buffers
            for (size_t j = 0; j < b.winIdx.size(); j++)
            {
                const auto &d = b.layout.descs[j];
                fastibs::packWindow(windows[b.winIdx[j]].seq,
                                    seqWords.data() + d.seqWordOff,
                                    invWords.data() + d.invWordOff);
            }
            std::vector<uint32_t> hitWords(b.layout.outWords, 0);
            std::vector<uint32_t> validWords(b.layout.outWords, 0);
            for (uint64_t w = 0; w < b.layout.outWords; w++)
                fastibs::encodeProbeWord(w, b.layout.descs.data(),
                                         (uint32_t)b.layout.descs.size(),
                                         seqWords.data(), invWords.data(), k,
                                         probe, hitWords[w], validWords[w]);
            for (size_t j = 0; j < b.winIdx.size(); j++)
            {
                const auto &d = b.layout.descs[j];
                fastibs::BitStats bs;
                RefStats &r = out[b.winIdx[j]];
                fastibs::BitStats::run(hitWords.data(), validWords.data(),
                                       d.outWordOff, d.nPos, bs,
                                       [&](uint64_t g) { r.gaps.push_back(g); });
                r.total = bs.totalKmers;
                r.observed = bs.observedKmers;
            }
        });
    return out;
}

// Random sequence with lowercase stretches and N runs of assorted lengths.
static std::string makeSequence(std::mt19937_64 &rng, size_t len, uint32_t k)
{
    static const char bases[] = "ACGT";
    static const char lower[] = "acgt";
    std::string s(len, 'A');
    bool lc = false;
    for (size_t i = 0; i < len; i++)
    {
        if (rng() % 97 == 0)
            lc = !lc;
        s[i] = (lc ? lower : bases)[rng() % 4];
    }
    // sprinkle N runs: lengths 1, k-1, k, 3k at random spots
    const size_t runs[] = {1, k - 1, k, 3 * k};
    for (size_t r : runs)
        if (len > r + 2)
        {
            const size_t at = rng() % (len - r);
            for (size_t i = 0; i < r; i++)
                s[at + i] = 'N';
        }
    return s;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: %s <kmc_db_path_prefix> [reference.fasta]\n", argv[0]);
        return 1;
    }

    std::mt19937_64 rng(20260831);

    // Part A: pipeline logic vs codec reference with a synthetic predicate
    // (no database involved) across many window shapes and both k values.
    for (uint32_t k : {25u, 31u, 32u})
    {
        auto pred = [](uint64_t code)
        { return (code * 0x9E3779B97F4A7C15ull >> 61) == 0; }; // ~12% hits

        // ids/sequences arranged like processReference: contigs chunked
        // into windows with k-1 overlap
        std::vector<std::string> ids;
        std::vector<std::string> seqs;
        for (size_t len : {200000ul, 50000ul, 1000ul, 100ul,
                           (unsigned long)k, (unsigned long)k - 1, 5ul, 33ul})
        {
            ids.push_back("contig" + std::to_string(seqs.size()));
            seqs.push_back(makeSequence(rng, len, k));
        }
        std::vector<SeqWindow> windows;
        chunkSequences(ids, seqs, 5000, k, windows);

        auto pipe = pipelineStats(windows, k, pred);
        size_t mismatches = 0;
        for (size_t i = 0; i < windows.size(); i++)
        {
            const auto ref = referenceStats(windows[i].seq, k, pred);
            if (!same(ref, pipe[i]))
            {
                if (!mismatches++)
                    printf("k=%u window %zu (%zu bases): total %llu/%llu "
                           "observed %llu/%llu gaps %zu/%zu\n",
                           k, i, windows[i].seq.size(),
                           (unsigned long long)ref.total,
                           (unsigned long long)pipe[i].total,
                           (unsigned long long)ref.observed,
                           (unsigned long long)pipe[i].observed,
                           ref.gaps.size(), pipe[i].gaps.size());
            }
        }
        CHECK(mismatches == 0, "pipeline/codec mismatch (synthetic predicate)");
        printf("k=%u synthetic predicate: %zu windows agree\n", k, windows.size());
    }

    // Part C: capacity-driven batch splitting must not change any result,
    // must respect the caps, and must reject a window no batch can hold.
    {
        const uint32_t k = 25;
        auto pred = [](uint64_t code)
        { return (code * 0x9E3779B97F4A7C15ull >> 62) == 0; };

        std::vector<std::string> ids;
        std::vector<std::string> seqs;
        for (size_t len : {30000ul, 7001ul, 500ul, 24ul, 64ul, 12345ul})
        {
            ids.push_back("c" + std::to_string(seqs.size()));
            seqs.push_back(makeSequence(rng, len, k));
        }
        std::vector<SeqWindow> windows;
        chunkSequences(ids, seqs, 3000, k, windows);
        std::vector<RefStats> ref(windows.size());
        for (size_t i = 0; i < windows.size(); i++)
            ref[i] = referenceStats(windows[i].seq, k, pred);

        // sweep tiny caps: many batches, single-window batches, one batch
        const fastibs::BatchCaps capSweep[] = {
            {3000 / 32 + 2, 3000 / 64 + 2, 3000 / 32 + 2, 1},   // 1 window/batch
            {200, 100, 200, 3},                                 // words bound
            {1u << 20, 1u << 20, 1u << 20, 2},                  // count bound
            {1u << 20, 1u << 20, 1u << 20, 1u << 20},           // everything fits
        };
        for (const auto &caps : capSweep)
        {
            size_t nBatches = 0;
            auto got = pipelineStatsBatched(windows, k, caps, pred, nBatches);
            size_t bad = 0;
            for (size_t i = 0; i < windows.size(); i++)
                if (!same(ref[i], got[i]))
                    bad++;
            CHECK(bad == 0, "batched pipeline differs from reference");
            printf("caps{seq=%zu,win=%zu}: %zu batches, %zu windows agree\n",
                   caps.seqWords, caps.windows, nBatches, windows.size());
        }

        // a window larger than any batch must throw, not overflow
        bool threw = false;
        try
        {
            size_t nBatches = 0;
            const fastibs::BatchCaps tiny{10, 10, 10, 4};
            pipelineStatsBatched(windows, k, tiny, pred, nBatches);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        CHECK(threw, "oversized window did not throw");
        printf("oversized window rejected: ok\n");
    }

    // Part B: same comparison against the real database probe — the exact
    // functor class the GPU kernel uses (probeOne via KmcRaIndex).
    const std::string db = argv[1];
    KmcRaIndex index(db);
    const uint32_t k = index.getParams().kmerLength;
    auto probe = [&](uint64_t code) { return index.probe(code); };

    // Random sequences exercise the miss path; pass a reference FASTA the
    // database was built from (e.g. the make_test_corpus.sh output) to
    // exercise hit-heavy windows too.
    std::vector<std::string> ids{"chrA", "chrB"};
    std::vector<std::string> seqs{makeSequence(rng, 3000000, k),
                                  makeSequence(rng, 123457, k)};
    if (argc > 2)
    {
        auto parsed = readSequenceFile(argv[2]);
        ids.insert(ids.end(), parsed.ids.begin(), parsed.ids.end());
        seqs.insert(seqs.end(), parsed.sequences.begin(), parsed.sequences.end());
    }
    std::vector<SeqWindow> windows;
    chunkSequences(ids, seqs, 50000, k, windows);

    auto pipe = pipelineStats(windows, k, probe);
    size_t mismatches = 0;
    uint64_t totalHits = 0;
    for (size_t i = 0; i < windows.size(); i++)
    {
        const auto ref = referenceStats(windows[i].seq, k, probe);
        totalHits += ref.observed;
        if (!same(ref, pipe[i]))
            mismatches++;
    }
    CHECK(mismatches == 0, "pipeline/codec mismatch (real database)");
    printf("k=%u real db: %zu windows agree (%llu observed k-mers)\n", k,
           windows.size(), (unsigned long long)totalHits);

    if (failures)
    {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL PHASE 2 HOST TESTS PASSED\n");
    return 0;
}
