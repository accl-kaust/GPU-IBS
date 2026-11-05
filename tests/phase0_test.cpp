// Phase 0 regression test: validates the rolling codec, CKmerAPI::from_int,
// the difference-array mapper, and the reworked FASTA/FASTQ reader against
// string-based legacy oracles.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a tests/phase0_test.cpp KMC/kmc_api/kmer_api.cpp \
//       KMC/kmc_api/mmer.cpp -lz -o phase0_test && ./phase0_test

#include <cassert>
#include <charconv>
#include <random>
#include <unordered_set>
#include <cstdio>

#include "../src/Utils.hpp"
#include "../KMC/kmc_api/kmer_api.h"

static mt19937_64 rng(20260830);

static string randomSequence(size_t len, bool withInvalid, bool withLower)
{
    static const char bases[] = "ACGT";
    string s(len, 'A');
    for (auto &c : s)
    {
        int r = rng() % 100;
        if (withInvalid && r < 5)
            c = "NnXY-"[rng() % 5];
        else
        {
            c = bases[rng() % 4];
            if (withLower && r >= 50)
                c = tolower(c);
        }
    }
    return s;
}

// ---- oracle: legacy string-based canonical extraction (per position) ----
static bool oracleAt(const string &upper, size_t i, size_t k, string &canonical)
{
    string kmer = upper.substr(i, k);
    if (kmer.find_first_not_of("ACGT") != string::npos)
        return false;
    string rc = reverseComplement(kmer);
    canonical = kmer > rc ? rc : kmer;
    return true;
}

static uint64_t packKmer(const string &kmer)
{
    uint64_t v = 0;
    for (char c : kmer)
        v = (v << 2) | (uint64_t)(c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : 3);
    return v;
}

static int failures = 0;
#define CHECK(cond, msg)                                        \
    do                                                          \
    {                                                           \
        if (!(cond))                                            \
        {                                                       \
            printf("FAIL: %s (line %d)\n", msg, __LINE__);      \
            failures++;                                         \
        }                                                       \
    } while (0)

static void testCodec(size_t k)
{
    KmerCodec codec(k);
    for (int trial = 0; trial < 60; trial++)
    {
        size_t len = rng() % 400; // includes len < k
        string seq = randomSequence(len, trial % 2, trial % 3 != 0);
        string upper = seq;
        transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

        vector<uint64_t> expected;
        vector<uint8_t> expectedValid;
        if (len >= k)
            for (size_t i = 0; i + k <= len; i++)
            {
                string canonical;
                bool ok = oracleAt(upper, i, k, canonical);
                expectedValid.push_back(ok);
                if (ok)
                    expected.push_back(packKmer(canonical));
            }

        vector<uint64_t> codes;
        codec.getCanonicalCodes(seq, codes);
        CHECK(codes == expected, "getCanonicalCodes mismatch vs oracle");

        vector<uint64_t> posCodes;
        vector<uint8_t> posValid;
        codec.encodePositions(seq, posCodes, posValid);
        CHECK(posValid == expectedValid, "encodePositions validity mismatch");
        vector<uint64_t> compacted;
        for (size_t i = 0; i < posCodes.size(); i++)
            if (posValid[i])
                compacted.push_back(posCodes[i]);
        CHECK(compacted == expected, "encodePositions codes mismatch");
    }
    printf("codec k=%zu: OK\n", k);
}

static void testFromInt(size_t k)
{
    for (int trial = 0; trial < 200; trial++)
    {
        string kmer = randomSequence(k, false, false);
        CKmerAPI a, b(k);
        CHECK(a.from_string(kmer), "from_string failed");
        CHECK(b.from_int(packKmer(kmer), k), "from_int failed");
        CHECK(a == b, "from_int != from_string");
        CHECK(b.to_string() == kmer, "from_int round-trip mismatch");
    }
    printf("from_int k=%zu: OK\n", k);
}

static void testMapperEquivalence(size_t k)
{
    for (int trial = 0; trial < 30; trial++)
    {
        size_t len = k + rng() % 300;
        string seq = randomSequence(len, true, true);
        string upper = seq;
        transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

        // fake membership: ~50% of encountered canonical k-mers
        unordered_set<uint64_t> member;
        KmerCodec codec(k);
        vector<uint64_t> codes;
        codec.getCanonicalCodes(seq, codes);
        for (auto c : codes)
            if (rng() % 2)
                member.insert(c);

        // legacy algorithm: k scattered increments per hit
        vector<short> legacyMapping(len, 0);
        for (size_t i = 0; i + k <= len; i++)
        {
            string canonical;
            if (!oracleAt(upper, i, k, canonical))
                continue;
            if (member.count(packKmer(canonical)))
                for (size_t j = 0; j < k; j++)
                    legacyMapping[i + j] += 1;
        }
        string legacyStr;
        for (auto m : legacyMapping)
            legacyStr += to_string(m) + ",";
        if (!legacyStr.empty())
            legacyStr.pop_back();

        // optimized algorithm: difference array + prefix pass + to_chars
        vector<uint64_t> posCodes;
        vector<uint8_t> posValid;
        codec.encodePositions(seq, posCodes, posValid);
        vector<int> diff(len + 1, 0);
        for (size_t i = 0; i < posCodes.size(); i++)
        {
            if (!posValid[i])
                continue;
            if (member.count(posCodes[i]))
            {
                diff[i] += 1;
                diff[i + k] -= 1;
            }
        }
        string optStr;
        optStr.reserve(2 * len);
        char buf[12];
        int coverage = 0;
        for (size_t i = 0; i < len; i++)
        {
            coverage += diff[i];
            auto res = to_chars(buf, buf + sizeof(buf), coverage);
            optStr.append(buf, res.ptr);
            optStr.push_back(',');
        }
        if (!optStr.empty())
            optStr.pop_back();

        CHECK(optStr == legacyStr, "mapper serialization mismatch");
    }
    printf("mapper equivalence k=%zu: OK\n", k);
}

// ---- oracle: legacy stringstream-based reader semantics ----
static ParseResult legacyParse(const string &data, FileType fileType)
{
    ParseResult result;
    stringstream ss(data);
    string currentID, line;
    stringstream currentSeq;
    bool isQualityLine = false;
    while (getline(ss, line))
    {
        if (line.empty())
            continue;
        if (fileType == FileType::FASTQ && line[0] == '+')
        {
            isQualityLine = true;
            continue;
        }
        if (isQualityLine)
        {
            isQualityLine = false;
            continue;
        }
        char idChar = (fileType == FileType::FASTA ? '>' : '@');
        if (line[0] == idChar)
        {
            if (!currentSeq.str().empty())
            {
                result.ids.push_back(currentID);
                result.sequences.push_back(currentSeq.str());
                currentSeq.str(string());
            }
            currentID = line.substr(1);
        }
        else
            currentSeq << line;
    }
    if (!currentSeq.str().empty())
    {
        result.ids.push_back(currentID);
        result.sequences.push_back(currentSeq.str());
    }
    return result;
}

static void writeFile(const string &path, const string &content)
{
    ofstream f(path, ios::binary);
    f << content;
}

static void testReader()
{
    vector<pair<string, FileType>> cases = {
        {">c1\nACGTACGT\nACGT\n>c2\nTTTT\n", FileType::FASTA},
        {">c1\nACGT", FileType::FASTA},          // no trailing newline
        {">c1\n\nACGT\n\n>c2\n\nGGGG\n", FileType::FASTA}, // empty lines
        {">only_header\n>c2\nACGT\n", FileType::FASTA},    // empty record
        {"@r1\nACGT\n+\nIIII\n@r2\nGGTT\n+\n!!!!\n", FileType::FASTQ},
        {"@r1\nACGTNNGT\n+r1\nIIIIIIII\n", FileType::FASTQ},
    };
    int idx = 0;
    for (auto &[data, type] : cases)
    {
        string path = "test_reader_tmp_" + to_string(idx++) + (type == FileType::FASTA ? ".fasta" : ".fastq");
        writeFile(path, data);
        auto got = readSequenceFile(path);
        auto expected = legacyParse(data, type);
        CHECK(got.ids == expected.ids, "reader ids mismatch");
        CHECK(got.sequences == expected.sequences, "reader sequences mismatch");
        remove(path.c_str());
    }
    printf("reader: OK\n");
}

static void testGzip()
{
    string data = ">c1\n";
    data += randomSequence(100000, true, true);
    data += "\n>c2\n";
    data += randomSequence(50000, false, false);
    data += "\n";
    writeFile("test_gzip_tmp.fasta", data);
    if (system("gzip -kf test_gzip_tmp.fasta") != 0)
    {
        printf("gzip: SKIPPED (no gzip binary)\n");
        remove("test_gzip_tmp.fasta");
        return;
    }
    string decompressed = decompressGzip("test_gzip_tmp.fasta.gz");
    CHECK(decompressed == data, "gzip round-trip mismatch");
    auto viaGz = readSequenceFile("test_gzip_tmp.fasta.gz", true);
    auto viaPlain = readSequenceFile("test_gzip_tmp.fasta");
    CHECK(viaGz.ids == viaPlain.ids && viaGz.sequences == viaPlain.sequences,
          "gz vs plain parse mismatch");
    remove("test_gzip_tmp.fasta");
    remove("test_gzip_tmp.fasta.gz");
    printf("gzip: OK\n");
}

int main()
{
    for (size_t k : {4, 15, 25, 31, 32})
    {
        testCodec(k);
        testFromInt(k);
        testMapperEquivalence(k);
    }
    testReader();
    testGzip();

    if (failures)
    {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL TESTS PASSED\n");
    return 0;
}
