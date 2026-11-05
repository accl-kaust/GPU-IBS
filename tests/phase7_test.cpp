// Input-handling host-side regression test (the release checks of the
// three tools, src/CliInput.hpp, and the k > 32 routing):
//   (1) <windowSize> parsing accepts exactly what std::stoi accepted
//       without trailing characters, in [1, INT_MAX], and the W > k bound
//       matches what chunkSequences needs: W <= k throws instead of
//       looping, W = k + 1 terminates with windows stepping by 1, and a
//       production window gives the same windows as the original loop,
//   (2) the reference-file filter (.fasta, .fa, .fna, each optionally
//       .gz) and the output stems derived from it (unchanged for .fasta),
//   (3) the gzip reader: a compressed reference parses to the same
//       contigs as the plain file; a truncated one throws,
//   (4) the k > 32 routing: needsStringPath is k > 32 for every k KMC
//       accepts; given databases, KmcRaIndex::readHeader agrees with the
//       KMC API's header, and opening a k > 32 database switches the
//       process to the string path (the index image refuses it) while a
//       k <= 32 database leaves the accelerated paths selected.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a -DBOOST_ALLOW_DEPRECATED_HEADERS \
//       -DBOOST_BIND_GLOBAL_PLACEHOLDERS -DBOOST_TIMER_ENABLE_DEPRECATED \
//       tests/phase7_test.cpp KMC/kmc_api/kmc_file.cpp \
//       KMC/kmc_api/kmer_api.cpp KMC/kmc_api/mmer.cpp -lz -lpthread -o phase7_test
// Run (without FASTIBS_LEGACY in the environment):
//   ./phase7_test [k<=32_db_path_prefix [k>32_db_path_prefix]]
// The database checks of (4) run only for the prefixes given; the k > 32
// one must come last (opening it switches the whole process).

#include <climits>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>
#include <zlib.h>

#include "../src/CliInput.hpp"
#include "../src/KmerDatabase.hpp"
#include "../src/gpu/KmcRaIndex.hpp"

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

// The original window loop (Utils.hpp before the guard), for W > k.
static vector<pair<size_t, size_t>> originalWindows(size_t len, size_t W, size_t k)
{
    vector<pair<size_t, size_t>> out;
    for (size_t j = 0; j < len; j += W - k)
        out.push_back({j, j + W > len ? len : j + W});
    return out;
}

static void testWindowSize()
{
    struct Case
    {
        const char *text;
        bool ok;
        int value;
    };
    const Case cases[] = {
        {"50000", true, 50000},     {"10000", true, 10000},   {"+50000", true, 50000},
        {" 50000", true, 50000},    {"050000", true, 50000},  {"1", true, 1},
        {"2147483647", true, INT_MAX},
        {"0", false, 0},            {"-5", false, 0},         {"50000abc", false, 0},
        {"5e4", false, 0},          {"", false, 0},           {" ", false, 0},
        {"50000 ", false, 0},       {"2147483648", false, 0}, {"99999999999", false, 0},
        {"0x10", false, 0},         {"abc", false, 0},
    };
    for (const Case &c : cases)
    {
        int v = -1;
        const bool ok = cli::parseWindowSize(c.text, v);
        char msg[96];
        snprintf(msg, sizeof msg, "parseWindowSize(\"%s\")", c.text);
        CHECK(ok == c.ok, msg);
        if (ok && c.ok)
        {
            CHECK(v == c.value, msg);
            // what the original parse gave for the accepted forms
            CHECK(v == std::stoi(c.text), msg);
        }
    }

    for (unsigned k : {1u, 21u, 25u, 31u, 32u, 35u, 63u})
    {
        CHECK(!cli::windowFitsK(k - 1, k), "W = k - 1 must be rejected");
        CHECK(!cli::windowFitsK(k, k), "W = k must be rejected");
        CHECK(cli::windowFitsK(k + 1, k), "W = k + 1 must be accepted");
        CHECK(cli::windowFitsK(50000, k), "W = 50000 must be accepted");
    }

    // chunkSequences: the guard, and unchanged windows otherwise.
    const vector<string> ids = {"c1", "c2", "c3"};
    const vector<string> seqs = {string(123457, 'A'), string(31, 'C'), string(25, 'G')};
    vector<SeqWindow> chunks;
    for (size_t W : {24, 25})
    {
        bool threw = false;
        try
        {
            chunkSequences(ids, seqs, W, 25, chunks);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        CHECK(threw, "chunkSequences must throw for W <= k");
    }
    for (size_t W : {26, 100, 10000, 50000})
    {
        chunkSequences(ids, seqs, W, 25, chunks);
        size_t at = 0;
        bool same = true;
        for (size_t c = 0; c < seqs.size(); c++)
            for (const auto &w : originalWindows(seqs[c].size(), W, 25))
            {
                same = same && at < chunks.size() && chunks[at].id == &ids[c] &&
                       chunks[at].start == w.first && chunks[at].end == w.second &&
                       chunks[at].seq.size() == w.second - w.first;
                at++;
            }
        same = same && at == chunks.size();
        CHECK(same, "chunkSequences windows differ from the original loop");
    }
    printf("window size: %zu parse cases, W > k bound, chunkSequences guard and windows OK\n",
           sizeof cases / sizeof cases[0]);
}

static void testReferenceFiles()
{
    const char *yes[] = {"test_genome.fasta", "x.fa", "x.fna", "x.fasta.gz", "x.fa.gz",
                         "x.fna.gz", "a.b.fasta"};
    const char *no[] = {"x.fastq", "x.gz", "x.fasta.fai", "x.txt", "fasta", ".gz",
                        "x.FASTA", "x.fasta.gz.tmp", "x.fasta.bgz"};
    for (const char *f : yes)
        CHECK(cli::isReferenceFile(f), f);
    for (const char *f : no)
        CHECK(!cli::isReferenceFile(f), f);
    // fastibsmapper's output stem: unchanged for *.fasta (the original's
    // path.stem()), the same stem for the compressed file
    auto stem = [](const string &f)
    { return std::filesystem::path(cli::stripGz(f)).stem().string(); };
    CHECK(stem("test_genome.fasta") == std::filesystem::path("test_genome.fasta").stem().string(),
          "mapper stem of a .fasta changed");
    CHECK(stem("test_genome.fasta.gz") == "test_genome", "mapper stem of a .fasta.gz");
    CHECK(stem("a.b.fasta") == "a.b", "mapper stem of a dotted name");
    CHECK(stem("x.fna.gz") == "x", "mapper stem of a .fna.gz");
    printf("reference files: %zu accepted, %zu skipped, output stems OK\n",
           sizeof yes / sizeof yes[0], sizeof no / sizeof no[0]);
}

static void testGzipReader()
{
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("phase7_" + std::to_string(getpid()));
    fs::create_directories(dir);
    string fasta;
    for (int c = 0; c < 5; c++)
    {
        fasta += ">contig" + std::to_string(c) + " description\n";
        for (int l = 0; l < 200 + 37 * c; l++)
            fasta += string(60, "ACGTNacgt"[(c + l) % 9]) + "\n";
    }
    const string plain = (dir / "r.fasta").string(), gz = (dir / "r.fasta.gz").string(),
                 cut = (dir / "cut.fasta.gz").string();
    {
        std::ofstream(plain, std::ios::binary) << fasta;
        gzFile f = gzopen(gz.c_str(), "wb");
        gzwrite(f, fasta.data(), (unsigned)fasta.size());
        gzclose(f);
        std::ifstream in(gz, std::ios::binary);
        string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::ofstream(cut, std::ios::binary) << bytes.substr(0, bytes.size() / 2);
    }
    const ParseResult a = readSequenceFile(plain, false);
    const ParseResult b = readSequenceFile(gz, true);
    CHECK(a.ids == b.ids && a.sequences == b.sequences, "gzip reference parses differently");
    CHECK(a.ids.size() == 5, "contig count");
    bool threw = false;
    try
    {
        readSequenceFile(cut, true);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    CHECK(threw, "a truncated gzip reference must throw");
    fs::remove_all(dir);
    printf("gzip reader: compressed == plain (%zu contigs), truncated stream rejected\n",
           a.ids.size());
}

static void testRouting(const vector<string> &dbs)
{
    for (uint k = 1; k <= 256; k++)
        CHECK(KmerDatabase::needsStringPath(k) == (k > 32), "needsStringPath(k) != (k > 32)");
    CHECK(!KmerDatabase::useLegacyPath(), "string path selected before any database "
                                          "(FASTIBS_LEGACY set?)");
    for (const string &prefix : dbs)
    {
        const KmcRaIndex::Header h = KmcRaIndex::readHeader(prefix);
        CKMCFile api;
        CKMCFileInfo info;
        CHECK(api.OpenForListing(prefix.c_str()), "KMC API cannot open the database");
        api.Info(info);
        CHECK(h.kmerLength == info.kmer_length, "readHeader k differs from the KMC API");
        CHECK(h.totalKmers == info.total_kmers, "readHeader k-mer count differs from the KMC API");
        CHECK(h.lutPrefixLength == info.lut_prefix_length, "readHeader p differs from the KMC API");
        const bool longK = KmerDatabase::needsStringPath(h.kmerLength);
        bool imageThrew = false;
        try
        {
            KmcRaIndex image(prefix, true);
        }
        catch (const std::runtime_error &)
        {
            imageThrew = true;
        }
        CHECK(imageThrew == longK, "the index image must refuse exactly the k > 32 databases");
        KmerDatabase db(prefix);
        CHECK(db.getKmerSize() == h.kmerLength, "KmerDatabase k");
        CHECK(KmerDatabase::useLegacyPath() == longK,
              longK ? "a k > 32 database must select the string path"
                    : "a k <= 32 database must leave the accelerated paths selected");
        CHECK(!KmerDatabase::cpuBatchedApplies() || !longK,
              "the batched probe must not apply to a k > 32 database");
        printf("routing: %s (k=%u, %llu k-mers): %s\n", prefix.c_str(), h.kmerLength,
               (unsigned long long)h.totalKmers,
               longK ? "string path (CPU-BASE)" : "accelerated paths");
    }
}

int main(int argc, char **argv)
{
    testWindowSize();
    testReferenceFiles();
    testGzipReader();
    testRouting(vector<string>(argv + 1, argv + argc));

    if (failures)
    {
        printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("\nALL PHASE 7 HOST TESTS PASSED\n");
    return 0;
}
