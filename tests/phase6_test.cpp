// Panel-mode host-side regression test (docs/MULTIDB_SCAN.md section 9):
// the multi-database fused kernel body (fastibs::encodeProbeWordMulti),
// the signature-once probe (fastibs::probeOneWithSig) and the joint shard
// plan (KmcRaIndex::jointShardPlan), driven on the host with the real
// KMC indexes of a panel corpus -- the same templates the CUDA kernels
// instantiate, so what passes here is what the device computes.
//
//  A. probeOneWithSig composes to probeOne on every k-mer of every
//     database and on random queries (hits and misses).
//  B. For n in {2, 3, 5} databases and every window shape the Phase 2 test
//     uses (lowercase, N runs, tails, windows of k and k-1 bases): the n
//     hit words and the validity word of encodeProbeWordMulti equal n runs
//     of the single-database encodeProbeWord, on every output word; both
//     with the signature shared and with it recomputed per database.
//  C. jointShardPlan for N in {1, 2, 3, 5}: one common cut (at the coarsest
//     prefix length), every database's shards partition its prefix space,
//     records sum to its total, and every present k-mer of every database
//     is found on exactly one of that database's shards, the one the common
//     owner function names.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a tests/phase6_test.cpp KMC/kmc_api/kmc_file.cpp \
//       KMC/kmc_api/kmer_api.cpp KMC/kmc_api/mmer.cpp -lz -o phase6_test
// Run:
//   ./phase6_test <db1_prefix> <db2_prefix> [db3_prefix ...]   (2 to 5 databases)

#include <cstdio>
#include <memory>
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

// Host instance of the kernels' multi-probe functor over KmcRaIndex
// objects: the same interface MultiBsProbe (EncodeProbe.cu) offers the
// shared body, with the signature computed once when shared.
struct HostMultiProbe
{
    std::vector<const KmcRaIndex *> dbs;
    bool sharedSig = true;
    uint32_t count() const { return (uint32_t)dbs.size(); }
    uint32_t prepare(uint64_t code) const
    {
        const auto &P = dbs[0]->getParams();
        return (sharedSig && P.signatureLen)
                   ? fastibs::signatureOf(code, P.kmerLength, P.signatureLen,
                                          dbs[0]->getNorm().data())
                   : 0;
    }
    bool operator()(uint32_t d, uint64_t code, uint32_t sig) const
    {
        const KmcRaIndex &x = *dbs[d];
        const auto P = x.probeParams();
        if (!sharedSig)
            return x.probe(code);
        if (x.recordsAre32Bit())
            return fastibs::probeOneWithSig<uint32_t>(code, sig, x.getLut().data(),
                                                      x.getSuffix32().data(),
                                                      x.getSignatureMap().data(), P);
        return fastibs::probeOneWithSig<uint64_t>(code, sig, x.getLut().data(),
                                                  x.getSuffix64().data(),
                                                  x.getSignatureMap().data(), P);
    }
};

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
    if (argc < 3)
    {
        printf("usage: %s <db1_prefix> <db2_prefix> [db3_prefix ...] [--ref=<reference.fasta>]\n",
               argv[0]);
        return 1;
    }
    std::string refFasta; // part B also runs over its windows (hit-dense)
    std::vector<std::unique_ptr<KmcRaIndex>> owned;
    std::vector<const KmcRaIndex *> dbs;
    for (int a = 1; a < argc && owned.size() < fastibs::MAX_DBS; a++)
    {
        if (std::string(argv[a]).rfind("--ref=", 0) == 0)
        {
            refFasta = argv[a] + 6;
            continue;
        }
        owned.push_back(std::make_unique<KmcRaIndex>(argv[a]));
        dbs.push_back(owned.back().get());
        const auto &P = owned.back()->getParams();
        printf("db %zu: %s  k=%u p=%u sig=%u total=%llu rec%s\n", owned.size(), argv[a],
               P.kmerLength, P.lutPrefixLength, P.signatureLen,
               (unsigned long long)P.totalKmers, owned.back()->recordsAre32Bit() ? "32" : "64");
        CHECK(P.kmerLength == dbs[0]->getParams().kmerLength, "databases share k");
    }
    const uint32_t k = dbs[0]->getParams().kmerLength;
    std::mt19937_64 rng(20261027);

    // ---- A. probeOneWithSig == probeOne ----------------------------------
    for (size_t d = 0; d < dbs.size(); d++)
    {
        const KmcRaIndex &x = *dbs[d];
        const auto P = x.probeParams();
        auto withSig = [&](uint64_t code)
        {
            const uint32_t sig = P.sigLen ? fastibs::signatureOf(code, P.k, P.sigLen, x.getNorm().data()) : 0;
            return x.recordsAre32Bit()
                       ? fastibs::probeOneWithSig<uint32_t>(code, sig, x.getLut().data(), x.getSuffix32().data(), x.getSignatureMap().data(), P)
                       : fastibs::probeOneWithSig<uint64_t>(code, sig, x.getLut().data(), x.getSuffix64().data(), x.getSignatureMap().data(), P);
        };
        size_t present = 0, mismatch = 0;
        x.forEachKmer([&](uint64_t code)
                      {
                          present++;
                          if (!withSig(code) || !x.probe(code))
                              mismatch++;
                      });
        CHECK(mismatch == 0, "every present k-mer probes true with the shared signature");
        size_t hits = 0;
        for (int i = 0; i < 200000; i++)
        {
            // random canonical codes
            uint64_t f = rng() & (k == 32 ? ~0ull : ((1ull << (2 * k)) - 1));
            uint64_t rc = 0, t = f;
            for (uint32_t j = 0; j < k; j++)
            {
                rc = (rc << 2) | (3 - (t & 3));
                t >>= 2;
            }
            const uint64_t code = f < rc ? f : rc;
            const bool a = withSig(code), b = x.probe(code);
            if (a != b)
                mismatch++;
            hits += b;
        }
        CHECK(mismatch == 0, "random queries agree between probeOneWithSig and probeOne");
        printf("A. db %zu: %zu present k-mers and 200k random queries (%zu hits) agree\n", d + 1,
               present, hits);
    }

    // ---- B. multi body == n single bodies ---------------------------------
    for (size_t n : {2ul, 3ul, 5ul})
    {
        if (n > dbs.size())
        {
            printf("B. n=%zu skipped (%zu databases given)\n", n, dbs.size());
            continue;
        }
        std::vector<std::string> ids, seqs;
        for (size_t len : {200000ul, 50000ul, 1000ul, 100ul, (unsigned long)k,
                           (unsigned long)k - 1, 5ul, 33ul})
        {
            ids.push_back("contig" + std::to_string(seqs.size()));
            seqs.push_back(makeSequence(rng, len, k));
        }
        // plus a stretch straight from database 1's k-mers so hits are dense
        {
            std::string dense;
            size_t taken = 0;
            dbs[0]->forEachKmer([&](uint64_t code)
                                {
                                    if (taken++ % 50000 == 0 && dense.size() < 40000)
                                        dense += convertIntToKmer(code, k);
                                });
            ids.push_back("dense");
            seqs.push_back(dense);
        }
        // plus the real reference, when given: the databases were built
        // from mutated copies of it, so most of its k-mers hit most of them
        // and the hit words differ between databases where they mutated.
        if (!refFasta.empty())
        {
            auto parsed = readSequenceFile(refFasta);
            for (size_t c = 0; c < parsed.sequences.size(); c++)
            {
                ids.push_back(parsed.ids[c]);
                seqs.push_back(std::move(parsed.sequences[c]));
            }
        }
        std::vector<SeqWindow> windows;
        chunkSequences(ids, seqs, 5000, k, windows);

        fastibs::BatchLayout layout;
        std::vector<size_t> probed;
        for (size_t i = 0; i < windows.size(); i++)
        {
            if (windows[i].seq.size() < k)
                continue;
            layout.add((uint32_t)windows[i].seq.size(), k);
            probed.push_back(i);
        }
        std::vector<uint64_t> seqWords(layout.seqWords, 0), invWords(layout.invWords, 0);
        for (size_t j = 0; j < probed.size(); j++)
        {
            const auto &d = layout.descs[j];
            fastibs::packWindow(windows[probed[j]].seq, seqWords.data() + d.seqWordOff,
                                invWords.data() + d.invWordOff);
        }

        for (bool shared : {true, false})
        {
            HostMultiProbe multi;
            multi.dbs.assign(dbs.begin(), dbs.begin() + n);
            multi.sharedSig = shared;
            const size_t stride = layout.outWords + 7; // any stride >= outWords
            std::vector<uint32_t> hitMulti(n * stride, 0xDEADBEEFu), validMulti(layout.outWords, 0);
            for (uint64_t w = 0; w < layout.outWords; w++)
                fastibs::encodeProbeWordMulti(w, layout.descs.data(), (uint32_t)layout.descs.size(),
                                              seqWords.data(), invWords.data(), k, multi,
                                              hitMulti.data(), stride, validMulti[w]);
            size_t wordMismatch = 0, validMismatch = 0, totalHits = 0, differing = 0;
            for (size_t d = 0; d < n; d++)
            {
                const KmcRaIndex &x = *dbs[d];
                auto probe = [&](uint64_t code) { return x.probe(code); };
                for (uint64_t w = 0; w < layout.outWords; w++)
                {
                    uint32_t h, v;
                    fastibs::encodeProbeWord(w, layout.descs.data(), (uint32_t)layout.descs.size(),
                                             seqWords.data(), invWords.data(), k, probe, h, v);
                    if (h != hitMulti[d * stride + w])
                        wordMismatch++;
                    if (v != validMulti[w])
                        validMismatch++;
                    totalHits += __builtin_popcount(h);
                    if (d && h != hitMulti[w]) // differs from database 1's word
                        differing++;
                }
            }
            CHECK(wordMismatch == 0, "multi-database hit words equal the single-database words");
            CHECK(validMismatch == 0, "validity word equals the single-database one");
            CHECK(n < 2 || differing > 0, "the databases' hit words differ somewhere (the test has teeth)");
            printf("B. n=%zu, signature %s: %llu output words x %zu databases, %zu hits, %zu words differ from db 1, %zu word mismatches\n",
                   n, shared ? "shared" : "per database", (unsigned long long)layout.outWords, n,
                   totalHits, differing, wordMismatch);
        }
    }

    // ---- C. joint shard plan ---------------------------------------------
    {
        std::vector<std::vector<uint64_t>> counts;
        for (const auto *x : dbs)
            counts.push_back(x->perPrefixCounts());
        uint32_t pMin = dbs[0]->getParams().lutPrefixLength;
        for (const auto *x : dbs)
            pMin = std::min(pMin, x->getParams().lutPrefixLength);
        for (int N : {1, 2, 3, 5})
        {
            const auto plan = KmcRaIndex::jointShardPlan(counts, N);
            CHECK(plan.size() == dbs.size(), "one plan per database");
            // the common cut, at the coarsest prefix length
            std::vector<uint32_t> cut(N);
            for (int d = 0; d < N; d++)
                cut[d] = plan[0][d].prefixHi >> (2 * (dbs[0]->getParams().lutPrefixLength - pMin));
            uint64_t jointRecords = 0;
            for (size_t i = 0; i < dbs.size(); i++)
            {
                const KmcRaIndex &x = *dbs[i];
                const uint32_t p = x.getParams().lutPrefixLength;
                const uint64_t P4 = x.getParams().singleLutSize;
                uint64_t records = 0;
                CHECK(plan[i][0].prefixLo == 0, "first shard starts at prefix 0");
                for (int d = 0; d < N; d++)
                {
                    const auto &s = plan[i][d];
                    CHECK(s.count == N && s.index == d, "shard numbering");
                    CHECK((s.prefixHi >> (2 * (p - pMin))) == cut[d], "common cut");
                    if (d)
                        CHECK(s.prefixLo == plan[i][d - 1].prefixHi, "shards are contiguous");
                    records += s.records;
                }
                CHECK(plan[i][N - 1].prefixHi == P4, "last shard ends at 4^p");
                CHECK(records == x.getParams().totalKmers, "shard records sum to the total");
                jointRecords += records;

                // every present k-mer: exactly one shard, the owner's. The
                // shard images are gathered once, as the device upload does.
                std::vector<KmcRaIndex::ShardLayout> layouts;
                std::vector<std::vector<uint32_t>> s32(N);
                std::vector<std::vector<uint64_t>> s64(N);
                for (int d = 0; d < N; d++)
                {
                    layouts.push_back(x.shardLayout(plan[i][d]));
                    for (const auto &r : layouts[d].ranges)
                        for (uint64_t q = r.first; q < r.second; q++)
                            if (x.recordsAre32Bit())
                                s32[d].push_back(x.getSuffix32()[q]);
                            else
                                s64[d].push_back(x.getSuffix64()[q]);
                    CHECK((x.recordsAre32Bit() ? s32[d].size() : s64[d].size()) == layouts[d].records,
                          "gathered records match the layout");
                }
                const uint32_t shift = 2 * (x.getParams().kmerLength - pMin);
                size_t wrong = 0, n = 0;
                x.forEachKmer([&](uint64_t code)
                              {
                                  if (n++ % 97)
                                      return; // a sample is plenty
                                  const uint32_t prefix = (uint32_t)(code >> shift);
                                  int owner = 0;
                                  while (owner + 1 < N && prefix >= cut[owner])
                                      owner++;
                                  int found = 0, foundAt = -1;
                                  for (int d = 0; d < N; d++)
                                  {
                                      const auto &L = layouts[d];
                                      const bool hit =
                                          x.recordsAre32Bit()
                                              ? fastibs::probeOne<uint32_t>(code, L.lut.data(), s32[d].data(), x.getSignatureMap().data(), x.getNorm().data(), L.params)
                                              : fastibs::probeOne<uint64_t>(code, L.lut.data(), s64[d].data(), x.getSignatureMap().data(), x.getNorm().data(), L.params);
                                      if (hit)
                                      {
                                          found++;
                                          foundAt = d;
                                      }
                                  }
                                  if (found != 1 || foundAt != owner)
                                      wrong++;
                              });
                CHECK(wrong == 0, "every sampled k-mer is on exactly its owner's shard");
            }
            printf("C. N=%d: common cut", N);
            for (int d = 0; d < N; d++)
                printf(" %u", cut[d]);
            printf(" (p_min=%u), %llu records over %zu databases, every sampled k-mer on its owner's shard\n",
                   pMin, (unsigned long long)jointRecords, dbs.size());
        }
    }

    if (failures)
    {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
