#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include "CliInput.hpp"
#include "KmerDatabase.hpp"
#include "gpu/KmcRaIndex.hpp"
#include "gpu/ParallelEnum.hpp"

#ifdef FASTIBS_CUDA
#include <memory>
#include "gpu/DeviceKmerIndex.cuh"
#include "gpu/DeviceHashIndex.cuh"
#include "gpu/HashCore.hpp"
#endif

using namespace std;
namespace fs = filesystem;

void removeTrailingSlash(string &path)
{
    if (!path.empty() && path.back() == '/')
    {
        path.pop_back();
    }
}

// The original stream-and-probe through the KMC API (FASTIBS_LEGACY=1),
// kept as the correctness oracle of the two paths below.
uint64_t getIntersectionSize(KmerDatabase &db1, KmerDatabase &db2)
{
    // 64-bit: intersections of wheat-scale databases exceed 2^32
    uint64_t intersectionSize = 0;
    CKmerAPI kmer(db1.getKmerSize());
    while (db1.readNextKmer(kmer))
        if (db2.isKmer(kmer))
            intersectionSize += 1;
    return intersectionSize;
}

// Threads enumerating the smaller database: FASTIBS_THREADS=N (0, empty
// or non-numeric: the default), else the CPUs in the process's affinity
// mask — the SLURM allocation, not the node's hardware concurrency, as
// for the index repack (KmcRaIndex::usableCpus). Clamped to 1..256; 1 is
// the sequential walk of the original path.
static unsigned enumThreads()
{
    unsigned n = KmcRaIndex::usableCpus();
    if (const char *v = getenv("FASTIBS_THREADS"))
    {
        const long t = atol(v);
        if (t > 0)
            n = (unsigned)min<long>(t, 256);
    }
    return max(1u, min(n, 256u));
}

// After the enumeration: every k-mer of the small database must have
// reached a probe batch; anything else is a bug and the caller falls back
// to the next path rather than print a wrong count.
static void checkEnumerated(const fastibs::EnumCountStats &stats, const KmcRaIndex &small)
{
    if (stats.keys != small.getParams().totalKmers)
        throw std::runtime_error("enumeration delivered " + to_string(stats.keys) +
                                 " of " + to_string(small.getParams().totalKmers) +
                                 " k-mers to the probe");
}

// Keys per probe batch, per device: 32M (256 MB) per device for a
// single-threaded run, as before, shared out among the enumeration threads
// (each fills its own batches) down to 2M (16 MB), so the buffered total
// stays at about three times the single-threaded batch whatever the
// thread count. FASTIBS_INTERSECT_BATCH_M=N overrides (millions of keys).
static size_t batchKeysFor(unsigned threads, int nDev)
{
    const size_t chunk = max<size_t>(1ull << 20, (32ull << 20) / (size_t)max(1, nDev));
    size_t keys = max<size_t>(min<size_t>(chunk, 2ull << 20), chunk / threads);
    if (const char *v = getenv("FASTIBS_INTERSECT_BATCH_M"))
    {
        const long n = atol(v); // 0, empty or non-numeric: keep the default
        if (n > 0)
            keys = (size_t)n << 20;
    }
    return keys;
}

// One stderr line per run saying where the time went: when the probe calls
// fill the wall time the run is probe-bound, when the enumeration does it
// is host-bound. The probe seconds are summed over the consumers, so with
// several devices (or the CPU path's one consumer per thread) they can
// exceed the wall time; the "enumerated in" seconds run until the last
// producer finished and include the time producers waited on full queues.
static void logEnumStats(const char *tag, const fastibs::EnumCountStats &s)
{
    const double rate = s.wallSeconds > 0 ? (double)s.keys / s.wallSeconds / 1e6 : 0;
    const double share = s.wallSeconds > 0 ? 100.0 * s.probeSeconds / s.wallSeconds : 0;
    const ios::fmtflags flags = cerr.flags();
    const streamsize precision = cerr.precision();
    cerr << fixed << setprecision(2) << tag << " intersect: " << s.keys
         << " k-mers enumerated by " << s.producers << " thread(s) in " << s.enumSeconds
         << " s as " << s.batches << " batches of up to " << s.batchKeys << " keys; "
         << s.wallSeconds << " s wall (" << setprecision(0) << rate << " Mkeys/s), "
         << setprecision(2) << s.probeSeconds << " s inside the probe calls on "
         << s.consumers << " consumer(s) = " << setprecision(0) << share << "% of wall"
         << endl;
    cerr.flags(flags);
    cerr.precision(precision);
}

#ifdef FASTIBS_CUDA
// Phase 4 GPU path: the larger database becomes the device-resident probe
// structure; the smaller one's k-mers are enumerated straight from its
// on-disk index (no KMC listing pass) and probed in batches, with only the
// per-batch count crossing back — the batched-probe primitive applied to
// the all-pairs matrix workload.
//
// Phase 5, several devices: the larger database is SPLIT across them
// (DeviceKmerIndex::buildShards, one prefix-range shard per device). The
// enumerated k-mers are already on the host, so they are routed here to
// the shard holding their prefix — the same test the device applies —
// and every query crosses PCIe once, to the one device that searches it.
// The per-shard counts add up to the intersection because each k-mer is
// held by exactly one shard.
//
// The enumeration runs on FASTIBS_THREADS producer threads (ParallelEnum.hpp:
// each walks its own bucket ranges of the index and fills its own batches);
// one consumer thread per device feeds the batches to probeCountHost, which
// is synchronous, so the device probes while the producers keep filling.
// Single-threaded, the enumeration was the whole run (the 2e9 probes of the
// self-intersection are ~1.5 s of 17.6 s): the parallel walk makes the
// intersection probe-bound. Returns true on success.
static bool gpuIntersect(const string &smallPath, const string &bigPath,
                         uint64_t &count)
{
    const int avail = DeviceKmerIndex::deviceCount();
    if (avail == 0)
    {
        // stdout stays the count alone; diagnostics go to stderr.
        cerr << "[gpu] no CUDA device found, using CPU intersection (cudaGetDeviceCount: "
             << DeviceKmerIndex::deviceCountError() << ")" << endl;
        return false;
    }
    try
    {
        KmcRaIndex small(smallPath);
        const int nDev = KmerDatabase::devicesToUse(avail);
        // The big database's host copy is needed only for the upload; end
        // its lifetime before the long enumeration so peak host RAM during
        // the probe loop is just the small database (matters for all-pairs
        // runs packing many intersect processes on one node).
        // FASTIBS_PROBE=hash: probe a device hash set of the larger database
        // (the GPU-HASH backend of the scan) instead of its ported index —
        // the intersection is the workload where probing dominates, so it
        // is where a faster probe structure can repay its build.
        const char *probeEnv = getenv("FASTIBS_PROBE");
        const bool useHash = probeEnv && string(probeEnv) == "hash";
        double loadFactor = 0.6;
        if (const char *lf = getenv("FASTIBS_HASH_LF"))
            loadFactor = atof(lf);
        if (useHash && (loadFactor < 0.1 || loadFactor > 0.95))
        {
            cerr << "Error: FASTIBS_HASH_LF must be in [0.1, 0.95]" << endl;
            exit(1);
        }
        vector<unique_ptr<DeviceKmerIndex>> shards;
        vector<unique_ptr<DeviceHashIndex>> hashShards;
        vector<uint32_t> prefixCut; // shard d holds prefixes [cut[d], cut[d+1])
        uint32_t prefixShift = 0;
        {
            KmcRaIndex big(bigPath);
            if (big.getParams().kmerLength != small.getParams().kmerLength)
            {
                cerr << "Error: K-mer lengths do not match" << endl;
                exit(1);
            }
            if (nDev > 1)
                cerr << "[gpu] splitting the index across " << nDev << " of " << avail
                     << " visible devices" << endl;
            if (useHash)
            {
                cerr << "[gpu] probe backend: hash set, load factor " << loadFactor
                     << " (FASTIBS_PROBE=hash)" << endl;
                hashShards = DeviceHashIndex::buildShards(big, loadFactor, nDev);
            }
            else
            {
                shards = DeviceKmerIndex::buildShards(big, nDev);
                for (const auto &s : shards)
                    prefixCut.push_back(s->shard().prefixLo);
                prefixCut.push_back(shards.back()->shard().prefixHi);
                prefixShift = 2 * (big.getParams().kmerLength - big.getParams().lutPrefixLength);
            }
        }
        auto ownerOf = [&](uint64_t code)
        {
            if (useHash) // the hash set is split by a salted owner hash of the key
                return (int)fastibs::hashShardOf(code, (uint32_t)nDev);
            const uint64_t prefix = code >> prefixShift;
            int d = 0; // a handful of shards: a linear scan beats a search
            while (d + 1 < nDev && prefix >= prefixCut[d + 1])
                ++d;
            return d;
        };
        // probeCountHost serializes on the shard's staging buffers and sets
        // the calling thread's device itself, so one consumer per device.
        auto probeBatch = [&](int d, const vector<uint64_t> &keys) -> uint64_t
        {
            if (useHash)
                return hashShards[d]->probeCountHost(keys);
            return shards[d]->probeCountHost(keys);
        };

        const unsigned threads = enumThreads();
        const size_t batchKeys = batchKeysFor(threads, nDev);
        fastibs::EnumCountStats stats;
        count = fastibs::enumerateAndCount(small, threads, nDev, batchKeys, 1, ownerOf,
                                           probeBatch, &stats);
        logEnumStats("[gpu]", stats);
        checkEnumerated(stats, small);
        return true;
    }
    catch (const std::exception &e)
    {
        cerr << "[gpu] " << e.what() << " — falling back to CPU intersection"
             << endl;
        return false;
    }
}
#endif

// The CPU path, the same shape: the smaller database enumerated from its
// index by the same producers, the larger one held in host RAM by
// KmcRaIndex and probed by KmcRaIndex::probe (bit-exact with
// CKMCFile::CheckKmer; the oracle of the device kernels) — the host probe
// is read-only, so one consumer per thread. This is what the CPU-only build
// and FASTIBS_GPU=0 run, so the producer/consumer code is exercised
// without a device. Returns false (reason on stderr) when an index cannot
// be loaded; main then falls back to the KMC-API stream-and-probe.
static bool cpuIntersect(const string &smallPath, const string &bigPath,
                         uint64_t &count)
{
    try
    {
        KmcRaIndex small(smallPath);
        KmcRaIndex big(bigPath);
        if (big.getParams().kmerLength != small.getParams().kmerLength)
        {
            cerr << "Error: K-mer lengths do not match" << endl;
            exit(1);
        }
        const unsigned threads = enumThreads();
        const size_t batchKeys = batchKeysFor(threads, 1);
        auto probeBatch = [&](int, const vector<uint64_t> &keys)
        {
            uint64_t hits = 0;
            for (uint64_t code : keys)
                hits += big.probe(code);
            return hits;
        };
        fastibs::EnumCountStats stats;
        count = fastibs::enumerateAndCount(
            small, threads, 1, batchKeys, threads, [](uint64_t) { return 0; },
            probeBatch, &stats);
        logEnumStats("[cpu]", stats);
        checkEnumerated(stats, small);
        return true;
    }
    catch (const std::exception &e)
    {
        cerr << "[cpu] " << e.what() << " — falling back to the KMC API intersection"
             << endl;
        return false;
    }
}

// The count, alone on stdout; a failed write (closed pipe, full disk) is an
// error, not a silent empty result.
static int printCount(uint64_t count)
{
    cout << count << endl;
    if (!cout)
    {
        cerr << "Error: writing the count to standard output failed" << endl;
        return 1;
    }
    return 0;
}

static int run(int argc, char *argv[])
{
    string kDB1Path, kDB2Path, database1, database2;
    if (cli::versionRequested(argc, argv))
    {
        cout << cli::versionLine("KDBIntersect") << endl;
        return 0;
    }
    const bool help = cli::helpRequested(argc, argv);
    if (argc != 3 || help)
    {
        cout << "\nK-mer Database Intersection Tool\n"
             << "----------------------------------\n"
             << "Usage:\n"
             << "  " << argv[0] << " <kDB1Path> <kDB2Path>\n\n"
             << "Arguments:\n"
             << "  <kDB1Path>     Path to the first KMC database directory\n"
             << "                 (e.g., /mnt/data/kmc_sets/db1)\n\n"
             << "  <kDB2Path>     Path to the second KMC database directory\n"
             << "                 (e.g., /mnt/data/kmc_sets/db2)\n\n"
             << "Description:\n"
             << "  This tool computes the intersection size between two KMC databases, i.e.,\n"
             << "  how many k-mers from the first database are present in the second.\n\n"
             << "Notes:\n"
             << "  - The databases must have the same k-mer size.\n"
             << "  - The tool compares the two KMC databases by k-mer count: the smaller one\n"
             << "    is enumerated and the larger one probed by random access (on a tie, the\n"
             << "    original rule: the .kmc_pre size, then the argument order).\n"
             << "  - The intersection size is output as an integer, which represents the\n"
             << "    number of common k-mers between the two databases.\n"
             << "  - k <= 32 takes the enumeration paths below; databases with k > 32 run the\n"
             << "    original KMC-API stream-and-probe.\n"
             << "  - GPU builds probe the larger database on the visible GPUs, split across\n"
             << "    them when there are several (FASTIBS_GPUS=N limits the count,\n"
             << "    FASTIBS_GPU=0 forces the CPU); the count is identical either way.\n"
             << "  - FASTIBS_PROBE=hash probes a device hash set of the larger database\n"
             << "    instead of its ported index (FASTIBS_HASH_LF: load factor, default 0.6);\n"
             << "    the count is identical, the table costs 8/lf bytes per k-mer.\n"
             << "  - The smaller database is enumerated on FASTIBS_THREADS threads (default:\n"
             << "    the CPUs in the process's affinity mask, at most 256; 1 is the sequential\n"
             << "    walk), on the GPU and on the CPU path alike; FASTIBS_LEGACY=1 runs the\n"
             << "    original KMC-API stream-and-probe instead. The count is identical in\n"
             << "    every case.\n"
             << "  - Exit status: 0 with the count on stdout; 1 on a bad command line or\n"
             << "    database (one error line on stderr).\n"
             << "  - --version prints the GPU-IBS version and build.\n\n"
             << "Example:\n"
             << "  " << argv[0] << " /mnt/data/kmc_sets/db1 /mnt/data/kmc_sets/db2\n\n";
        return help ? 0 : 1;
    }

    else
    {
        kDB1Path = argv[1];
        kDB2Path = argv[2];
    }

    removeTrailingSlash(kDB1Path);
    removeTrailingSlash(kDB2Path);

    const string dir1 = kDB1Path, dir2 = kDB2Path;
    if (!cli::resolveDatabase(dir1, kDB1Path) || !cli::resolveDatabase(dir2, kDB2Path))
        return 1;

    KmcRaIndex::Header h1, h2;
    try
    {
        h1 = KmcRaIndex::readHeader(kDB1Path);
        h2 = KmcRaIndex::readHeader(kDB2Path);
    }
    catch (const std::exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    if (h1.kmerLength != h2.kmerLength)
    {
        cerr << "Error: K-mer lengths do not match" << endl;
        return 1;
    }

    // The smaller database (by k-mer count, from the headers) is
    // enumerated and the larger one probed -- on the GPU path it becomes
    // the device-resident index. The original compared the .kmc_pre sizes
    // in whole MB, which follow the LUT's prefix length rather than the
    // record count (equal for databases built with the same settings, so
    // the argument order decided); that rule now only breaks ties. The
    // intersection is the same either way round.
    if (h1.totalKmers != h2.totalKmers)
    {
        if (h1.totalKmers > h2.totalKmers)
            swap(kDB1Path, kDB2Path);
    }
    else if (h1.fileBytes / 1024 / 1024 > h2.fileBytes / 1024 / 1024)
        swap(kDB1Path, kDB2Path);

    // k > 32: the index image (KmcRaIndex) holds a k-mer in 64 bits, so
    // both enumeration paths are skipped for the original one.
    if (KmerDatabase::needsStringPath(h1.kmerLength))
        KmerDatabase::routeToStringPath(h1.kmerLength);

    // FASTIBS_LEGACY=1 keeps the original stream-and-probe oracle; otherwise
    // the enumeration path, on the GPU by default when a device is present
    // (FASTIBS_GPU=0 forces the CPU) and on the CPU in a CPU-only build.
    const bool legacy = KmerDatabase::useLegacyPath();
#ifdef FASTIBS_CUDA
    if (KmerDatabase::gpuRequested() && !legacy)
    {
        uint64_t count = 0;
        if (gpuIntersect(kDB1Path, kDB2Path, count))
            return printCount(count);
    }
#endif
    if (!legacy)
    {
        uint64_t count = 0;
        if (cpuIntersect(kDB1Path, kDB2Path, count))
            return printCount(count);
    }

    KmerDatabase db1(kDB1Path, true); // open for listing
    KmerDatabase db2(kDB2Path, false); // open for random access

    if (db1.getKmerSize() != db2.getKmerSize())
    {
        cerr << "Error: K-mer lengths do not match" << endl;
        return 1;
    }
    auto intersectionSize = getIntersectionSize(db1, db2);
    return printCount(intersectionSize);
}

int main(int argc, char *argv[])
{
    // Anything the input checks did not anticipate ends in one error line
    // and exit status 1, not an uncaught exception's abort.
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
