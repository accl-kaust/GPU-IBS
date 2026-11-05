// Phase 4 host-side regression test:
//   (1) the mapper chunker partitions contig positions exactly (every
//       k-mer position in exactly one chunk),
//   (2) the GPU mapper flow — position-partitioned chunks through the
//       shared fused kernel body, difference-array accumulation, prefix
//       pass, serialization — produces byte-identical coverage strings to
//       the CPU mapper reference, on contigs with lowercase, N runs,
//       shorter-than-k, exactly-k and tail shapes,
//   (3) the GPU intersection flow — forEachKmer enumeration of one
//       database probed against the other's index — matches the CPU
//       stream-and-probe count (cross and self),
//   (4) the index split across devices (Phase 5): KmcRaIndex::shardPlan
//       partitions the prefix space with record counts summing to the
//       total, a single shard reproduces the on-disk arrays verbatim, and
//       every present k-mer probes true on exactly one shard's gathered
//       image (its plan owner) while random queries agree with the whole-
//       index probe — the same probeOne the sharded kernels run,
//   (5) the hash backend split the same way (hashShardOf partition, one
//       table per shard): exactly one shard holds each present key, and
//       the OR over shards equals the whole-index probe.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a -DBOOST_ALLOW_DEPRECATED_HEADERS \
//       -DBOOST_BIND_GLOBAL_PLACEHOLDERS -DBOOST_TIMER_ENABLE_DEPRECATED \
//       tests/phase4_test.cpp KMC/kmc_api/kmc_file.cpp \
//       KMC/kmc_api/kmer_api.cpp KMC/kmc_api/mmer.cpp -lz -lpthread -o phase4_test
// Run:
//   ./phase4_test <kmc_db_path_prefix> [second_db_path_prefix]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/KmerDatabase.hpp"
#include "../src/gpu/EncodeProbeCore.hpp"
#include "../src/gpu/HashCore.hpp"
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

// CPU mapper reference: codec + probe + difference array (the exact
// getMappingFromSequence logic, with an arbitrary probe functor).
template <typename ProbeFn>
static std::string referenceCoverage(const std::string &seq, uint32_t k,
                                     ProbeFn &&probe)
{
    std::vector<uint64_t> codes;
    std::vector<uint8_t> valid;
    KmerCodec(k).encodePositions(seq, codes, valid);
    std::vector<int> diff(seq.size() + 1, 0);
    for (size_t i = 0; i < codes.size(); i++)
    {
        if (!valid[i])
            continue;
        if (probe(codes[i]))
        {
            diff[i] += 1;
            diff[i + k] -= 1;
        }
    }
    return KmerDatabase::serializeCoverage(diff, seq.size());
}

// GPU mapper flow emulated on the host with the shared kernel body.
template <typename ProbeFn>
static std::vector<std::string> pipelineCoverage(const std::vector<std::string> &ids,
                                                 const std::vector<std::string> &seqs,
                                                 size_t W, uint32_t k,
                                                 ProbeFn &&probe)
{
    std::vector<SeqWindow> chunks;
    std::vector<size_t> chunkContig;
    chunkForMapping(ids, seqs, W, k, chunks, chunkContig);

    std::vector<std::vector<int>> diffs(seqs.size());
    for (size_t c = 0; c < seqs.size(); c++)
        diffs[c].assign(seqs[c].size() + 1, 0);

    fastibs::BatchLayout layout;
    for (const auto &w : chunks)
        layout.add((uint32_t)w.seq.size(), k);
    std::vector<uint64_t> seqWords(layout.seqWords, 0);
    std::vector<uint64_t> invWords(layout.invWords, 0);
    for (size_t j = 0; j < chunks.size(); j++)
    {
        const auto &d = layout.descs[j];
        fastibs::packWindow(chunks[j].seq, seqWords.data() + d.seqWordOff,
                            invWords.data() + d.invWordOff);
    }
    std::vector<uint32_t> hitWords(layout.outWords, 0);
    std::vector<uint32_t> validWords(layout.outWords, 0);
    for (uint64_t w = 0; w < layout.outWords; w++)
        fastibs::encodeProbeWord(w, layout.descs.data(),
                                 (uint32_t)layout.descs.size(), seqWords.data(),
                                 invWords.data(), k, probe, hitWords[w],
                                 validWords[w]);

    // consumer: hits -> difference arrays (same bit logic as the pipeline)
    for (size_t j = 0; j < chunks.size(); j++)
    {
        const auto &d = layout.descs[j];
        auto &diff = diffs[chunkContig[j]];
        const size_t base = chunks[j].start;
        for (uint32_t p = 0; p < d.nPos; p++)
        {
            const uint64_t word = d.outWordOff + p / 32;
            const uint32_t bit = p & 31;
            if (((validWords[word] >> bit) & (hitWords[word] >> bit)) & 1)
            {
                diff[base + p] += 1;
                diff[base + p + k] -= 1;
            }
        }
    }

    std::vector<std::string> out(seqs.size());
    for (size_t c = 0; c < seqs.size(); c++)
        out[c] = KmerDatabase::serializeCoverage(diffs[c], seqs[c].size());
    return out;
}

static std::string makeSequence(std::mt19937_64 &rng, size_t len, uint32_t k)
{
    static const char alpha[] = "ACGTacgt";
    std::string s(len, 'A');
    for (auto &c : s)
        c = alpha[rng() % 8];
    const size_t runs[] = {1, (size_t)k - 1, (size_t)k, 3 * (size_t)k};
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
        printf("usage: %s <kmc_db_path_prefix> [second_db_path_prefix]\n",
               argv[0]);
        return 1;
    }
    std::mt19937_64 rng(20260902);

    // --- 1 & 2: mapper chunking + coverage equivalence -------------------
    for (uint32_t k : {25u, 31u})
    {
        std::vector<std::string> ids;
        std::vector<std::string> seqs;
        for (size_t len : {500000ul, 4096ul, 100ul, (unsigned long)k,
                           (unsigned long)k - 1, 3ul, 12345ul})
        {
            ids.push_back("c" + std::to_string(seqs.size()));
            seqs.push_back(makeSequence(rng, len, k));
        }

        // chunker partition property, at several chunk sizes incl. tiny
        for (size_t W : {(size_t)4096, (size_t)k + 5, (size_t)100000})
        {
            std::vector<SeqWindow> chunks;
            std::vector<size_t> chunkContig;
            chunkForMapping(ids, seqs, std::max(W, (size_t)2 * k), k, chunks,
                            chunkContig);
            std::vector<std::vector<int>> seen(seqs.size());
            for (size_t c = 0; c < seqs.size(); c++)
                seen[c].assign(
                    seqs[c].size() >= k ? seqs[c].size() - k + 1 : 0, 0);
            for (size_t j = 0; j < chunks.size(); j++)
            {
                const size_t len = chunks[j].seq.size();
                for (size_t p = 0; p + k <= len; p++)
                    seen[chunkContig[j]][chunks[j].start + p]++;
            }
            size_t bad = 0;
            for (const auto &v : seen)
                for (int n : v)
                    if (n != 1)
                        bad++;
            CHECK(bad == 0, "mapper chunker does not partition positions");
        }

        auto pred = [](uint64_t code)
        { return (code * 0x9E3779B97F4A7C15ull >> 62) == 0; };
        auto got = pipelineCoverage(ids, seqs, 4096, k, pred);
        size_t mismatches = 0;
        for (size_t c = 0; c < seqs.size(); c++)
            if (got[c] != referenceCoverage(seqs[c], k, pred))
                mismatches++;
        CHECK(mismatches == 0, "pipeline coverage differs from reference");
        printf("k=%u mapper: chunker partitions exactly, %zu contigs byte-identical\n",
               k, seqs.size());
    }

    // --- mapper against the real database --------------------------------
    const std::string db = argv[1];
    KmcRaIndex index(db);
    const uint32_t k = index.getParams().kmerLength;
    auto probe = [&](uint64_t code) { return index.probe(code); };
    {
        std::vector<std::string> ids{"chrA", "chrB"};
        std::vector<std::string> seqs{makeSequence(rng, 300000, k),
                                      makeSequence(rng, 54321, k)};
        auto got = pipelineCoverage(ids, seqs, 65536, k, probe);
        size_t mismatches = 0;
        for (size_t c = 0; c < seqs.size(); c++)
            if (got[c] != referenceCoverage(seqs[c], k, probe))
                mismatches++;
        CHECK(mismatches == 0, "real-db coverage differs from reference");
        printf("k=%u mapper vs real db: %zu contigs byte-identical\n", k,
               seqs.size());
    }

    // --- 3: intersection equivalence --------------------------------------
    {
        // self-intersection: every enumerated k-mer must probe true
        uint64_t self = 0;
        index.forEachKmer([&](uint64_t code) { self += index.probe(code); });
        CHECK(self == index.getParams().totalKmers,
              "self-intersection != total k-mers");
        printf("self-intersection: %llu == total\n", (unsigned long long)self);

        if (argc > 2)
        {
            KmcRaIndex other(argv[2]);
            CHECK(other.getParams().kmerLength == k, "k mismatch");
            // GPU flow: enumerate `other`, probe `index`
            uint64_t viaEnum = 0;
            other.forEachKmer([&](uint64_t code)
                              { viaEnum += index.probe(code); });
            // CPU flow: KMC listing of `other`, probe `index`
            CKMCFile lister;
            if (!lister.OpenForListing(argv[2]))
            {
                printf("cannot open %s for listing\n", argv[2]);
                return 1;
            }
            CKmerAPI kmer(k);
            uint32 count;
            std::vector<uint64> words;
            uint64_t viaListing = 0;
            while (lister.ReadNextKmer(kmer, count))
            {
                kmer.to_long(words);
                viaListing += index.probe(words[0]);
            }
            CHECK(viaEnum == viaListing, "intersection counts differ");
            printf("cross-intersection: enumeration %llu == listing %llu\n",
                   (unsigned long long)viaEnum, (unsigned long long)viaListing);
        }
    }

    // --- 4: the index split across devices --------------------------------
    // Each shard's image is gathered on the host exactly as DeviceKmerIndex
    // gathers it on a device (shardLayout's LUT + record runs) and probed
    // with the shared probeOne under the shard's ProbeParams.
    {
        const auto &P = index.getParams();
        const uint64_t P4 = P.singleLutSize;
        const uint32_t shift = 2 * (P.kmerLength - P.lutPrefixLength);
        const uint64_t kmerMask = k == 32 ? ~0ull : (1ull << (2 * k)) - 1;
        const bool r32 = index.recordsAre32Bit();
        struct Image
        {
            KmcRaIndex::ShardLayout lay;
            std::vector<uint32_t> s32;
            std::vector<uint64_t> s64;
        };
        auto probeImage = [&](const Image &im, uint64_t code)
        {
            return r32 ? fastibs::probeOne<uint32_t>(code, im.lay.lut.data(), im.s32.data(),
                                                     index.getSignatureMap().data(),
                                                     index.getNorm().data(), im.lay.params)
                       : fastibs::probeOne<uint64_t>(code, im.lay.lut.data(), im.s64.data(),
                                                     index.getSignatureMap().data(),
                                                     index.getNorm().data(), im.lay.params);
        };
        for (int n : {1, 2, 3, 5})
        {
            const auto plan = index.shardPlan(n);
            CHECK((int)plan.size() == n, "shard plan size");
            bool contiguous = plan.front().prefixLo == 0 && plan.back().prefixHi == P4;
            uint64_t planRecords = 0;
            for (size_t i = 0; i < plan.size(); i++)
            {
                contiguous = contiguous && plan[i].prefixLo <= plan[i].prefixHi &&
                             (i == 0 || plan[i].prefixLo == plan[i - 1].prefixHi);
                planRecords += plan[i].records;
            }
            CHECK(contiguous, "shard plan does not partition the prefix space");
            CHECK(planRecords == P.totalKmers, "shard plan records do not sum to the total");

            std::vector<Image> images;
            for (const auto &sh : plan)
            {
                Image im;
                im.lay = index.shardLayout(sh);
                CHECK(im.lay.records == sh.records, "shard layout records != plan");
                for (const auto &r : im.lay.ranges)
                {
                    if (r32)
                        im.s32.insert(im.s32.end(), index.getSuffix32().begin() + r.first,
                                      index.getSuffix32().begin() + r.second);
                    else
                        im.s64.insert(im.s64.end(), index.getSuffix64().begin() + r.first,
                                      index.getSuffix64().begin() + r.second);
                }
                if (n == 1)
                {
                    // one device: the on-disk arrays, verbatim
                    CHECK(im.lay.lut == index.getLut(), "single shard LUT != on-disk LUT");
                    const bool sameSuffix =
                        r32 ? std::equal(im.s32.begin(), im.s32.end(),
                                         index.getSuffix32().begin(), index.getSuffix32().end())
                            : std::equal(im.s64.begin(), im.s64.end(),
                                         index.getSuffix64().begin(), index.getSuffix64().end());
                    CHECK(sameSuffix, "single shard suffix array != on-disk suffix array");
                }
                images.push_back(std::move(im));
            }

            // every present k-mer: found on exactly one shard, the plan's
            // owner of its prefix
            uint64_t bad = 0;
            index.forEachKmer([&](uint64_t code)
            {
                int hits = 0, owner = -1;
                for (size_t i = 0; i < images.size(); i++)
                    if (probeImage(images[i], code))
                    {
                        hits++;
                        owner = (int)i;
                    }
                const uint64_t prefix = code >> shift;
                bad += !(hits == 1 && prefix >= plan[owner].prefixLo &&
                         prefix < plan[owner].prefixHi);
            });
            CHECK(bad == 0, "a present k-mer is not held by exactly its owner shard");

            // random queries: OR over shards == whole-index probe, never
            // more than one shard hitting
            uint64_t disagree = 0, multi = 0, hitCount = 0;
            for (int i = 0; i < 200000; i++)
            {
                const uint64_t code = rng() & kmerMask;
                int hits = 0;
                for (const auto &im : images)
                    hits += probeImage(im, code);
                disagree += (hits > 0) != index.probe(code);
                multi += hits > 1;
                hitCount += hits > 0;
            }
            CHECK(disagree == 0, "sharded probe disagrees with the whole-index probe");
            CHECK(multi == 0, "a query hit on more than one shard");

            printf("index split %d-way:", n);
            for (const auto &sh : plan)
                printf(" [%u,%u) %.1f%%", sh.prefixLo, sh.prefixHi,
                       100.0 * sh.records / std::max<uint64_t>(P.totalKmers, 1));
            printf(" — present k-mers on exactly their owner shard, 200000 random agree (%llu hits)\n",
                   (unsigned long long)hitCount);
        }
    }

    // --- 5: hash backend split across devices -----------------------------
    {
        const uint64_t kmerMask = k == 32 ? ~0ull : (1ull << (2 * k)) - 1;
        for (uint32_t n : {1u, 3u})
        {
            std::vector<uint64_t> perShard(n, 0);
            index.forEachKmer([&](uint64_t code) { perShard[fastibs::hashShardOf(code, n)]++; });
            std::vector<fastibs::HashParams> HP(n);
            std::vector<std::vector<uint64_t>> tables(n);
            for (uint32_t d = 0; d < n; d++)
            {
                HP[d] = {fastibs::hashCapacityFor(perShard[d], 0.6), d, n};
                tables[d].assign(HP[d].capacity, fastibs::HASH_EMPTY);
            }
            uint64_t inserted = 0;
            index.forEachKmer([&](uint64_t code)
            {
                const uint32_t d = fastibs::hashShardOf(code, n);
                inserted += fastibs::hashInsertHost(code, tables[d].data(), HP[d]);
            });
            CHECK(inserted == index.getParams().totalKmers, "hash shard insert count mismatch");

            uint64_t bad = 0;
            index.forEachKmer([&](uint64_t code)
            {
                int hits = 0;
                for (uint32_t d = 0; d < n; d++)
                    hits += fastibs::hashContains(code, tables[d].data(), HP[d]);
                bad += hits != 1;
            });
            CHECK(bad == 0, "a present key is not held by exactly one hash shard");

            uint64_t disagree = 0;
            for (int i = 0; i < 200000; i++)
            {
                const uint64_t code = rng() & kmerMask;
                int hits = 0;
                for (uint32_t d = 0; d < n; d++)
                    hits += fastibs::hashContains(code, tables[d].data(), HP[d]);
                disagree += (hits > 0) != index.probe(code);
            }
            CHECK(disagree == 0, "sharded hash disagrees with the whole-index probe");
            printf("hash split %u-way: shard sizes", n);
            for (uint64_t c : perShard)
                printf(" %llu", (unsigned long long)c);
            printf(" — present keys on exactly one shard, 200000 random agree\n");
        }
    }

    if (failures)
    {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL PHASE 4 HOST TESTS PASSED\n");
    return 0;
}
