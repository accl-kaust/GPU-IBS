// Phase 1 host-side regression test: validates KmcRaIndex (direct
// .kmc_pre/.kmc_suf load + repack + Algorithm-1 probe with KMC2 signature
// bins) bit-exactly against the vendored CKMCFile on a real database.
// The GPU kernel mirrors KmcRaIndex::probe operation for operation, so this
// test is the CPU half of the Phase 1 validation gate. It also checks the
// batched CPU probe (CpuProbeBatched.hpp) against probe() query for query.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a tests/phase1_test.cpp KMC/kmc_api/kmc_file.cpp \
//       KMC/kmc_api/kmer_api.cpp KMC/kmc_api/mmer.cpp -lz -o phase1_test
// Run:
//   ./phase1_test <kmc_db_path_prefix>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/CpuProbeBatched.hpp"
#include "../src/KmerCodec.hpp"
#include "../src/gpu/KmcRaIndex.hpp"
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

    CKMCFile ra;
    if (!ra.OpenForRA(db))
    {
        printf("cannot open %s for RA\n", db.c_str());
        return 1;
    }
    CKMCFileInfo info;
    ra.Info(info);
    const uint32_t k = info.kmer_length;
    printf("db: k=%u lut_prefix=%u sig_len=%u counter_size=%u total=%llu\n",
           k, info.lut_prefix_length, info.signature_len, info.counter_size,
           (unsigned long long)info.total_kmers);

    KmcRaIndex index(db);
    const auto &P = index.getParams();
    CHECK(P.kmerLength == k, "kmer length mismatch");
    CHECK(P.totalKmers == info.total_kmers, "total_kmers mismatch");
    CHECK(P.lutPrefixLength == info.lut_prefix_length, "lut prefix mismatch");
    CHECK(P.signatureLen == info.signature_len, "signature len mismatch");
    printf("device footprint (counters stripped): %.2f MB\n",
           index.deviceFootprintBytes() / 1048576.0);
    printf("index read+repack: %.3f s (%u threads)\n", index.loadSeconds(),
           index.loadThreads());

    // 1) every k-mer present in the database must probe as a hit
    CKMCFile lister;
    if (!lister.OpenForListing(db))
    {
        printf("cannot open %s for listing\n", db.c_str());
        return 1;
    }
    CKmerAPI kmer(k);
    uint32 count;
    std::vector<uint64> words;
    uint64_t nListed = 0, nListedHit = 0;
    while (lister.ReadNextKmer(kmer, count))
    {
        kmer.to_long(words);
        const uint64_t code = words[0];
        nListed++;
        if (index.probe(code))
            nListedHit++;
        else if (nListed - nListedHit == 1)
            printf("first missed present k-mer: %s\n", kmer.to_string().c_str());
    }
    CHECK(nListed == info.total_kmers, "listing count mismatch");
    CHECK(nListedHit == nListed, "present k-mer probed as miss");
    printf("present k-mers: %llu/%llu hit\n",
           (unsigned long long)nListedHit, (unsigned long long)nListed);

    // 2) random k-mers: KmcRaIndex must agree with CKMCFile::IsKmer
    std::mt19937_64 rng(20260830);
    const uint64_t kmerMask = k == 32 ? ~0ull : (1ull << (2 * k)) - 1;
    uint64_t agree = 0, hits = 0;
    const int N = 2000000;
    CKmerAPI q(k);
    for (int i = 0; i < N; i++)
    {
        const uint64_t code = rng() & kmerMask;
        q.from_int(code, k);
        const bool a = ra.IsKmer(q);
        const bool b = index.probe(code);
        if (a == b)
            agree++;
        if (a)
            hits++;
        if (a != b && agree + 1 == (uint64_t)i)
            printf("first disagreement at code %llx: IsKmer=%d probe=%d\n",
                   (unsigned long long)code, (int)a, (int)b);
    }
    CHECK(agree == (uint64_t)N, "random k-mer disagreement");
    printf("random k-mers: %llu/%d agree (%llu hits)\n",
           (unsigned long long)agree, N, (unsigned long long)hits);

    // 3) codec-derived canonical codes from a random sequence (the actual
    //    query distribution of processReference)
    static const char bases[] = "ACGT";
    std::string seq(2000000, 'A');
    for (auto &c : seq)
        c = bases[rng() % 4];
    std::vector<uint64_t> codes;
    KmerCodec(k).getCanonicalCodes(seq, codes);
    agree = 0;
    hits = 0;
    for (const auto code : codes)
    {
        q.from_int(code, k);
        const bool a = ra.IsKmer(q);
        if (a == index.probe(code))
            agree++;
        if (a)
            hits++;
    }
    CHECK(agree == codes.size(), "codec-query disagreement");
    printf("codec queries: %llu/%zu agree (%llu hits)\n",
           (unsigned long long)agree, codes.size(), (unsigned long long)hits);

    // 4) the batched CPU probe (CpuProbeBatched.hpp, the CPU-BATCHED mode:
    //    K probes in flight per thread, prefetched level by level) must
    //    agree with probe() query for query — on the codec queries and on
    //    random k-mers, for every K from 1 to the maximum, and with a
    //    validity mask (the mapper's per-position encode) masking some
    //    positions out as misses.
    {
        std::vector<uint8_t> ref(codes.size()), bat(codes.size());
        for (size_t i = 0; i < codes.size(); i++)
            ref[i] = index.probe(codes[i]);
        std::vector<uint64_t> rnd(1000000);
        for (auto &c : rnd)
            c = rng() & kmerMask;
        std::vector<uint8_t> rndRef(rnd.size()), rndBat(rnd.size());
        for (size_t i = 0; i < rnd.size(); i++)
            rndRef[i] = index.probe(rnd[i]);
        for (unsigned K : {1u, 2u, 3u, 7u, 16u, 32u, 64u, fastibs::kCpuBatchMax, 2 * fastibs::kCpuBatchMax})
        {
            fastibs::probeBatched(index, codes.data(), nullptr, codes.size(), bat.data(), K);
            CHECK(bat == ref, "batched probe disagrees with probe() on codec queries");
            fastibs::probeBatched(index, rnd.data(), nullptr, rnd.size(), rndBat.data(), K);
            CHECK(rndBat == rndRef, "batched probe disagrees with probe() on random k-mers");
        }
        std::vector<uint8_t> valid(codes.size());
        for (size_t i = 0; i < valid.size(); i++)
            valid[i] = (i % 3) != 0 && (i % 101) != 0;
        fastibs::probeBatched(index, codes.data(), valid.data(), codes.size(), bat.data(), 16);
        uint64_t masked = 0, bad = 0;
        for (size_t i = 0; i < codes.size(); i++)
        {
            if (!valid[i])
                masked++;
            if (bat[i] != (valid[i] ? ref[i] : 0))
                bad++;
        }
        CHECK(bad == 0, "batched probe with validity mask disagrees");
        printf("batched probe: codec queries, random k-mers and %llu masked positions agree for K = 1..%u\n",
               (unsigned long long)masked, 2 * fastibs::kCpuBatchMax);
    }

    if (failures)
    {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL PHASE 1 HOST TESTS PASSED\n");
    return 0;
}
