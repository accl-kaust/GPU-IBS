#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <set>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <mutex>
#include <filesystem>
#include <charconv>
#include <cerrno>
#include <cstring>
#include <system_error>

#include <boost/progress.hpp>

#include "thread_pool.hpp"
#include "Window.hpp"
#include "Utils.hpp"
#include "../KMC/kmc_api/kmc_file.h"

#include <atomic>
#include <memory>
#include <thread>

// The host image of the index (CUDA-free) and the batched CPU probe over
// it: present in every build, the GPU path uploads the same arrays.
#include "gpu/KmcRaIndex.hpp"
#include "CpuProbeBatched.hpp"

#ifdef FASTIBS_CUDA
#include "gpu/DeviceKmerIndex.cuh"
#include "gpu/DeviceHashIndex.cuh"
#include "gpu/EncodeProbe.cuh"
#endif

#define CHUNK_SIZE 1000000 // defines the size of each sequence chunk when processing files

mutex m;

class KmerDatabase
{
    friend class KmerPanel; // panel mode drives N of these through one pass

public:
    uint getKmerSize() const
    {
        return kmerSize;
    }

    void printKMCInfo()
    {
        std::cout << "********** KMC Info **********\n";
        std::cout << "K-mer length: " << KMCInfo.kmer_length << '\n';
        std::cout << "Mode: " << KMCInfo.mode << '\n';
        std::cout << "Counter size: " << KMCInfo.counter_size << '\n';
        std::cout << "LUT prefix length: " << KMCInfo.lut_prefix_length << '\n';
        std::cout << "Signature length: " << KMCInfo.signature_len << '\n';
        std::cout << "Min count: " << KMCInfo.min_count << '\n';
        std::cout << "Max count: " << KMCInfo.max_count << '\n';
        std::cout << "Both strands: " << KMCInfo.both_strands << '\n';
        std::cout << "Total k-mers: " << KMCInfo.total_kmers << '\n';
    }

    KmerDatabase(string sourcePath, bool listing = false) : sourcePath(sourcePath)
    {
        if (listing)
        {
            if (!KMCDatabase.OpenForListing(sourcePath.c_str()))
            {
                std::cerr << "Error: Could not open KMC database " << sourcePath << "\n";
                exit(1);
            }
            KMCDatabase.Info(KMCInfo);
            raOpened = true; // listing mode never upgrades to RA
        }
        else
        {
            // Read only the header here (a transient listing-mode open —
            // cheap). The random-access open copies the entire suffix
            // array into host RAM, which GPU-probed runs never need, so
            // it is deferred until a CPU probe path actually asks for it
            // (ensureRaOpen).
            CKMCFile header;
            if (!header.OpenForListing(sourcePath.c_str()))
            {
                std::cerr << "Error: Could not open KMC database " << sourcePath << "\n";
                exit(1);
            }
            header.Info(KMCInfo);
        }
        kmerSize = KMCInfo.kmer_length;
        // printKMCInfo(KMCInfo);
        // The rolling codec, the host index image and the GPU index hold a
        // k-mer in one 64-bit word; a longer k takes the original string
        // path, which handles any k KMC accepts.
        if (needsStringPath(kmerSize))
            routeToStringPath(kmerSize);
    }

    // Whether a database of this k must take the original string path
    // (CPU-BASE): every accelerated path packs a k-mer into 64 bits.
    static bool needsStringPath(uint k)
    {
        return k > 32;
    }

    // Selects the string path for the whole process (every database of a
    // run shares k) and says so once; the GPU and --probe options are then
    // ignored, as under FASTIBS_LEGACY=1.
    static void routeToStringPath(uint k)
    {
        if (stringPathForced())
            return;
        stringPathForced() = true;
        std::cerr << "[fastibs] k=" << k
                  << " > 32: using the original string path (CPU-BASE); the GPU, rolling-codec "
                     "and batched-probe paths need k <= 32, so GPU and --probe options are ignored"
                  << std::endl;
    }

    // Opens the host-RAM random-access view on first use by a CPU probe
    // path (CPU stats, legacy oracle, mapper, intersect). Must be called
    // before concurrent probing starts.
    void ensureRaOpen()
    {
        std::lock_guard<std::mutex> lk(raOpenMutex);
        if (raOpened)
            return;
        if (!KMCDatabase.OpenForRA(sourcePath.c_str()))
        {
            std::cerr << "Error: Could not open KMC database " << sourcePath
                      << " for random access\n";
            exit(1);
        }
        raOpened = true;
    }

    // Opens the host image of the index (KmcRaIndex: counter-stripped,
    // native-integer records, read straight from the on-disk files — the
    // arrays the GPU path uploads) on first use by the batched CPU probe,
    // which reads it instead of the KMC random-access view. Must be called
    // before concurrent probing starts. 8 bytes per k-mer when the suffix
    // exceeds 4 bytes (the RA view holds suffix + counter bytes).
    void ensureHostIndex()
    {
        std::lock_guard<std::mutex> lk(raOpenMutex);
        if (hostIndex)
            return;
        try
        {
            hostIndex = std::make_unique<KmcRaIndex>(sourcePath);
        }
        catch (const std::exception &e)
        {
            // The loader reports a missing/truncated/unsupported file by
            // throwing; same outcome as a failed KMC open.
            std::cerr << "Error: Could not open KMC database for the batched probe: "
                      << e.what() << "\n";
            exit(1);
        }
        const double gb = hostIndex->loadBytes() / 1e9;
        char msg[160];
        snprintf(msg, sizeof msg,
                 "[cpu] index read+repack: %.3f s (%u threads, %.1f GB, %.2f GB/s)",
                 hostIndex->loadSeconds(), hostIndex->loadThreads(), gb,
                 gb / std::max(hostIndex->loadSeconds(), 1e-9));
        cout << msg << endl;
    }

    bool isKmer(CKmerAPI &kmer)
    {
        ensureRaOpen();
        return KMCDatabase.IsKmer(kmer);
    }

    bool readNextKmer(CKmerAPI &kmer)
    {
        uint32 count;
        return KMCDatabase.ReadNextKmer(kmer, count);
    }

    bool restartListing()
    {
        return KMCDatabase.RestartListing();
    }

    // Runtime switch: FASTIBS_LEGACY=1 selects the original string-based
    // path (CPU-BASE, the correctness oracle); default is the rolling-codec
    // path (CPU-OPT).
    static bool legacyRequested()
    {
        static const bool legacy = []
        {
            const char *v = getenv("FASTIBS_LEGACY");
            return v && *v && string(v) != "0";
        }();
        return legacy;
    }

    // Set once a database with k > 32 is opened (routeToStringPath).
    static bool &stringPathForced()
    {
        static bool forced = false;
        return forced;
    }

    // The string path runs when asked for (FASTIBS_LEGACY=1) or when k
    // leaves no other choice.
    static bool useLegacyPath()
    {
        return legacyRequested() || stringPathForced();
    }

    // GPU/pipeline knobs, settable from the CLI (--gpu/--cpu, --batch-mb,
    // --streams) before construction; CLI overrides the FASTIBS_GPU env
    // switch. Present in every build so the CLI parses uniformly; ignored
    // (with a warning from the CLI) when compiled without CUDA.
    struct GpuOptions
    {
        int enable = -1;   // -1 auto (env/default), 0 forced off, 1 forced on
        int batchMB = 128; // raw sequence bases per batch, in MB
        int streams = 3;   // pipeline depth (slots/streams)
        bool sync = false; // FASTIBS_GPU_SYNC=1: Phase 1 per-window path
        // Phase 3 ablation knobs
        bool sortQueries = false;     // --sort-queries: CUB sort between K1/K2
        bool probeHash = false;       // --probe=hash: hash-set backend
        double hashLoadFactor = 0.6;  // --hash-lf=F
        bool pinLut = false;          // --pin-lut: L2-persist the prefix LUT
        // Devices to use: --gpus=N; 0 = unset (FASTIBS_GPUS=N, else every
        // visible device). The index is SPLIT across them (Phase 5): device
        // d holds shard d and probes only its k-mers.
        int gpus = 0; // --gpus=N|all
        // --no-peer: host-staged routing between the devices even where
        // they have peer access (EncodeProbeEngine::forceStaged).
        bool noPeer = false;
        // --probe=batched: the K-way software-pipelined CPU probe over the
        // host index image (CPU-BATCHED) instead of the one-at-a-time KMC
        // API probe (CPU-OPT). A CPU-path knob; the CLI makes it imply --cpu.
        bool cpuBatched = false;
    };
    static GpuOptions &gpuOptions()
    {
        static GpuOptions o;
        return o;
    }

    // Size of the CPU thread pool (window tasks, packing, statistics).
    // FASTIBS_THREADS=N overrides the default, which stays
    // hardware_concurrency() -- the whole node's CPU count, as every recorded
    // baseline used -- so that a cpuset-limited job can be given exactly its
    // cores (16-core baselines) or a full socket (FASTIBS_THREADS=128).
    static unsigned poolThreads()
    {
        if (const char *v = getenv("FASTIBS_THREADS"))
        {
            const long n = atol(v);
            if (n > 0)
                return (unsigned)n;
        }
        return std::thread::hardware_concurrency();
    }

    // CPU probe backend: --probe=batched or FASTIBS_CPU_PROBE=batched
    // selects the K-way software-pipelined probe over the host index image
    // (CPU-BATCHED, CpuProbeBatched.hpp); the default stays the
    // one-at-a-time KMC API probe (CPU-OPT). Only consulted when the CPU
    // path probes and the legacy oracle is not selected.
    static bool cpuBatchedRequested()
    {
        if (gpuOptions().cpuBatched)
            return true;
        static const bool envBatched = []
        {
            const char *v = getenv("FASTIBS_CPU_PROBE");
            if (!v || !*v)
                return false;
            if (string(v) == "batched")
                return true;
            cerr << "Warning: FASTIBS_CPU_PROBE=" << v
                 << " is not a CPU probe mode (only \"batched\" is); using the default CPU-OPT probe"
                 << endl;
            return false;
        }();
        return envBatched;
    }

    // Whether the batched probe applies: requested, and the legacy oracle
    // (FASTIBS_LEGACY=1, the string-based path with its own probe loop)
    // not selected — said once when the oracle discards the request. A
    // GPU-probed run leaves it unused as well (the callers check that).
    static bool cpuBatchedApplies()
    {
        if (!cpuBatchedRequested())
            return false;
        if (stringPathForced())
            return false; // routeToStringPath already said the option is ignored
        if (useLegacyPath())
        {
            static const bool warned = []
            {
                cerr << "Warning: FASTIBS_LEGACY=1 selects the string-based CPU-BASE path; "
                        "--probe=batched / FASTIBS_CPU_PROBE=batched is ignored"
                     << endl;
                return true;
            }();
            (void)warned;
            return false;
        }
        return true;
    }

    // Probes kept in flight per worker thread by the batched probe:
    // FASTIBS_CPU_BATCH=K (default 16; clamped to 1..kCpuBatchMax; 0,
    // empty or non-numeric keeps the default).
    static unsigned cpuBatchK()
    {
        static const unsigned K = []
        {
            long n = 16;
            if (const char *v = getenv("FASTIBS_CPU_BATCH"))
                if (atol(v) > 0)
                    n = atol(v);
            return (unsigned)std::min<long>(n, fastibs::kCpuBatchMax);
        }();
        return K;
    }

    // Number of devices a GPU run uses out of the `available` visible ones:
    // --gpus=N, else FASTIBS_GPUS=N, else all of them. Also KDBIntersect's
    // rule (it has no options). CUDA_VISIBLE_DEVICES selects which ones.
    static int devicesToUse(int available)
    {
        int n = gpuOptions().gpus;
        if (n <= 0)
            if (const char *v = getenv("FASTIBS_GPUS"))
                n = atoi(v); // 0, empty or non-numeric: all
        if (n <= 0 || n > available)
            n = available;
        return n;
    }

#ifdef FASTIBS_CUDA
    // GPU probing defaults to on when built with CUDA and a device is
    // present; FASTIBS_GPU=0 (or --cpu) disables it at runtime.
    static bool gpuRequested()
    {
        if (gpuOptions().enable != -1)
            return gpuOptions().enable == 1;
        const char *v = getenv("FASTIBS_GPU");
        return !(v && *v && string(v) == "0");
    }

    static bool gpuSyncRequested()
    {
        if (gpuOptions().sync)
            return true;
        const char *v = getenv("FASTIBS_GPU_SYNC");
        return v && *v && string(v) != "0";
    }

    bool gpuActive() const
    {
        return !gpuIndexes.empty() || !gpuHashIndexes.empty();
    }

    // --gpu asks for the GPU path; with no usable device the run still
    // completes on the CPU (same output), but the request is not silently
    // dropped.
    static void warnGpuForcedWithoutDevice()
    {
        if (gpuOptions().enable == 1)
            cerr << "Warning: --gpu was given but no CUDA device is usable; running on the CPU"
                 << endl;
    }

    // plan (panel mode): split this database with the given shards, one
    // per device, instead of its own balanced plan — every database of a
    // round shares one cut (KmcRaIndex::jointShardPlan). The ported index
    // only; the panel refuses the hash backend before getting here.
    void initGpuIndex(const std::vector<KmcRaIndex::Shard> *plan = nullptr)
    {
        try
        {
            const int avail = DeviceKmerIndex::deviceCount();
            if (avail == 0)
            {
                cout << "[gpu] no CUDA device found, using CPU path (cudaGetDeviceCount: "
                     << DeviceKmerIndex::deviceCountError() << ")" << endl;
                warnGpuForcedWithoutDevice();
                return;
            }
            const int nDev = plan ? (int)plan->size() : devicesToUse(avail);
            // Re-read the on-disk index directly (KmcRaIndex); this host
            // copy is transient — freed once the device shards exist (the
            // hostIndex member is the batched CPU probe's, never both).
            KmcRaIndex image(sourcePath);
            {
                // GB and GB/s make a cold read (hundreds of MB/s from
                // storage) distinguishable from a warm one (GB/s from the
                // page cache) at a glance; the gate scripts print this line.
                const double gb = image.loadBytes() / 1e9;
                char msg[160];
                snprintf(msg, sizeof msg,
                         "[gpu] index read+repack: %.3f s (%u threads, %.1f GB, %.2f GB/s)",
                         image.loadSeconds(), image.loadThreads(), gb,
                         gb / std::max(image.loadSeconds(), 1e-9));
                cout << msg << endl;
            }
            // The hash backend needs the streamed pipeline; the Phase 1
            // sync path probes through the ported index only.
            const bool useHash = gpuOptions().probeHash && !gpuSyncRequested() && !plan;
            if (gpuOptions().probeHash && gpuSyncRequested())
                cout << "[gpu] --probe=hash is unsupported with --sync; "
                        "using the ported index"
                     << endl;

            // Phase 5: the index is SPLIT across the devices, not replicated.
            // Device d holds shard d — for the ported index a prefix range
            // of every signature bin (KmcRaIndex::shardPlan, balanced by
            // record count), for the hash backend the keys its owner hash
            // selects. In the streamed pipeline a batch goes to ONE device,
            // which encodes it and routes each query to the device holding
            // its shard (EncodeProbeEngine); each device holds ~1/N of the
            // index, so a database that does not fit one GPU fits N of
            // them. One device gets the whole index, uploaded verbatim as
            // before.
            const uint64_t total = image.getParams().totalKmers;
            const double perDevice =
                useHash ? 8.0 * total / gpuOptions().hashLoadFactor / nDev
                        : image.deviceFootprintBytes() / (double)nDev;
            if (nDev > 1)
            {
                char msg[200];
                snprintf(msg, sizeof msg,
                         "[gpu] splitting the index across %d of %d visible devices, ~%.2f GB per device",
                         nDev, avail, perDevice / 1e9);
                cout << msg << endl;
            }
            for (int d = 0; d < nDev; d++)
            {
                // Pre-flight: the upload would fail anyway, but this names
                // the knob to turn.
                const size_t freeBytes = DeviceKmerIndex::freeMemory(d);
                if (freeBytes && perDevice > (double)freeBytes)
                {
                    char msg[240];
                    snprintf(msg, sizeof msg,
                             "[gpu] device %d: its shard (~%.2f GB) exceeds the %.2f GB free on the device; "
                             "use more devices (--gpus / FASTIBS_GPUS / CUDA_VISIBLE_DEVICES)",
                             d, perDevice / 1e9, freeBytes / 1e9);
                    cout << msg << endl;
                }
            }
            if (useHash)
                gpuHashIndexes = DeviceHashIndex::buildShards(
                    image, gpuOptions().hashLoadFactor, nDev);
            else if (plan)
                gpuIndexes = DeviceKmerIndex::buildShards(image, *plan);
            else
                gpuIndexes = DeviceKmerIndex::buildShards(image, nDev);
        }
        catch (const std::exception &e)
        {
            cerr << "[gpu] " << e.what() << " — falling back to CPU path" << endl;
            gpuIndexes.clear();
            gpuHashIndexes.clear();
        }
    }

    // Frees this database's device image(s) (panel mode: a round's images
    // make room for the next round's). The next scan re-uploads lazily.
    void releaseGpuIndex()
    {
        gpuIndexes.clear();
        gpuHashIndexes.clear();
        gpuInitAttempted = false;
    }

    const std::vector<std::unique_ptr<DeviceKmerIndex>> &deviceShards() const
    {
        return gpuIndexes;
    }
#endif

    // Closes the host-RAM random-access view (its copy of the suffix array
    // is the host-memory cost of a CPU-probed database; panel mode on the
    // CPU closes a round's databases before opening the next round's).
    void releaseRa()
    {
        std::lock_guard<std::mutex> lk(raOpenMutex);
        if (!raOpened)
            return;
        KMCDatabase.Close();
        raOpened = false;
    }

    // Panel mode on the CPU: one window's statistics against several
    // databases from ONE pass of the rolling codec. Per database it is the
    // single-database probe loop of getStatsFromSequence over the same
    // codes, so each out[i] is what dbs[i]->getStatsFromSequence(window)
    // returns. Every database must have its random-access view open.
    static void getStatsFromSequenceMulti(const SeqWindow &window,
                                          const vector<KmerDatabase *> &dbs,
                                          Window *out)
    {
        const uint kmerSize = dbs[0]->kmerSize;
        vector<uint64_t> codes;
        KmerCodec(kmerSize).getCanonicalCodes(window.seq, codes);
        CKmerAPI KMCKmer(kmerSize);
        for (size_t i = 0; i < dbs.size(); i++)
        {
            Window &stats = out[i];
            stats = Window();
            stats.id = *window.id;
            stats.start = window.start;
            stats.end = window.end;
            stats.totalKmers = codes.size();
            CKMCFile &db = dbs[i]->KMCDatabase;
            int gapSize = 0;
            for (const auto code : codes)
            {
                KMCKmer.from_int(code, kmerSize);
                if (!db.IsKmer(KMCKmer))
                {
                    gapSize += 1;
                }
                else
                {
                    stats.observedKmers += 1;
                    if (gapSize > 0)
                    {
                        stats.addVariationOriginal(gapSize, int(kmerSize));
                        gapSize = 0;
                    }
                }
            }
            if (gapSize > 0)
                stats.addVariationOriginal(gapSize, kmerSize);
        }
    }

    Window getStatsFromSequence(const SeqWindow &window)
    {
        if (useLegacyPath())
            return getStatsFromSequenceLegacy(window);

        Window stats;
        stats.id = *window.id;
        stats.start = window.start;
        stats.end = window.end;
        const string_view sequence = window.seq;

        vector<uint64_t> codes;
        KmerCodec(kmerSize).getCanonicalCodes(sequence, codes);
        stats.totalKmers = codes.size();

#ifdef FASTIBS_CUDA
        if (!gpuIndexes.empty())
        {
            // Batched path: probe the whole window on the device(s), then
            // run the existing run-length statistics over the returned
            // bitvector. With the index split, the codes are already on
            // the host here, so each goes only to the shard holding its
            // prefix (the same test the streamed engine applies on the
            // device) and the answers are scattered back by index.
            vector<uint8_t> hits(codes.size(), 0);
            if (gpuIndexes.size() == 1)
                gpuIndexes[0]->probeHost(codes, hits);
            else
            {
                const auto P0 = gpuIndexes[0]->view().params;
                const uint32_t shift = 2 * (P0.k - P0.p);
                vector<vector<uint64_t>> perShard(gpuIndexes.size());
                vector<vector<uint32_t>> perIdx(gpuIndexes.size());
                for (size_t i = 0; i < codes.size(); i++)
                {
                    const uint64_t prefix = codes[i] >> shift;
                    size_t d = 0;
                    while (d + 1 < gpuIndexes.size() && prefix >= gpuIndexes[d]->shard().prefixHi)
                        d++;
                    perShard[d].push_back(codes[i]);
                    perIdx[d].push_back((uint32_t)i);
                }
                vector<uint8_t> shardHits;
                for (size_t d = 0; d < gpuIndexes.size(); d++)
                {
                    gpuIndexes[d]->probeHost(perShard[d], shardHits);
                    for (size_t j = 0; j < shardHits.size(); j++)
                        hits[perIdx[d][j]] = shardHits[j];
                }
            }
            int gapSize = 0;
            for (const auto hit : hits)
            {
                if (!hit)
                {
                    gapSize += 1;
                }
                else
                {
                    stats.observedKmers += 1;
                    if (gapSize > 0)
                    {
                        stats.addVariationOriginal(gapSize, int(kmerSize));
                        gapSize = 0;
                    }
                }
            }
            if (gapSize > 0)
                stats.addVariationOriginal(gapSize, kmerSize);
            return stats;
        }
#endif

        if (hostIndex)
        {
            // CPU-BATCHED: K probes in flight per thread over the host
            // image (the image is loaded only when this mode is selected),
            // then the run-length statistics over the hit flags, as on the
            // GPU-sync path above.
            vector<uint8_t> hits(codes.size());
            fastibs::probeBatched(*hostIndex, codes.data(), nullptr, codes.size(),
                                  hits.data(), cpuBatchK());
            int gapSize = 0;
            for (const auto hit : hits)
            {
                if (!hit)
                {
                    gapSize += 1;
                }
                else
                {
                    stats.observedKmers += 1;
                    if (gapSize > 0)
                    {
                        stats.addVariationOriginal(gapSize, int(kmerSize));
                        gapSize = 0;
                    }
                }
            }
            if (gapSize > 0)
                stats.addVariationOriginal(gapSize, kmerSize);
            return stats;
        }

        int gapSize = 0;
        CKmerAPI KMCKmer(kmerSize);
        for (const auto code : codes)
        {
            KMCKmer.from_int(code, kmerSize);
            if (!KMCDatabase.IsKmer(KMCKmer))
            {
                gapSize += 1;
            }
            else
            {
                stats.observedKmers += 1;
                if (gapSize > 0)
                {
                    stats.addVariationOriginal(gapSize, int(kmerSize));
                    gapSize = 0;
                }
            }
        }
        if (gapSize > 0)
            stats.addVariationOriginal(gapSize, kmerSize);

        return stats;
    }

    Window getStatsFromSequenceLegacy(const SeqWindow &window)
    {
        Window stats;
        stats.id = *window.id;
        stats.start = window.start;
        stats.end = window.end;
        string sequence(window.seq); // legacy extraction upper-cases in place

        auto kmers = getCanonicalKmersLegacy(sequence, kmerSize);
        stats.totalKmers = kmers.size();

        int gapSize = 0;
        CKmerAPI KMCKmer;
        for (const auto &kmer : kmers)
        {
            KMCKmer.from_string(kmer);
            if (!KMCDatabase.IsKmer(KMCKmer))
            {
                gapSize += 1;
            }
            else
            {
                stats.observedKmers += 1;
                if (gapSize > 0)
                {
                    stats.addVariationOriginal(gapSize, int(kmerSize));
                    gapSize = 0;
                }
            }
        }
        if (gapSize > 0)
            stats.addVariationOriginal(gapSize, kmerSize);

        return stats;
    }

    // Probes one position-partitioned chunk of a contig on the CPU and
    // accumulates hits into that contig's difference array (+1 at the hit
    // position, -1 at position+k). Used by the chunk-parallel CPU mapper;
    // the arithmetic is identical to the GPU consumer's, and the adds are
    // atomic for the same reason (a chunk's -1 entries can land in the
    // next chunk's range).
    void accumulateMappingChunk(const SeqWindow &chunk, vector<int> &diff)
    {
        vector<uint64_t> codes;
        vector<uint8_t> valid;
        KmerCodec(kmerSize).encodePositions(chunk.seq, codes, valid);
        if (hostIndex)
        {
            // CPU-BATCHED: the validity mask keeps the unspecified codes
            // of non-ACGT positions out of the probe groups.
            vector<uint8_t> hits(codes.size());
            fastibs::probeBatched(*hostIndex, codes.data(), valid.data(), codes.size(),
                                  hits.data(), cpuBatchK());
            for (size_t i = 0; i < codes.size(); i++)
            {
                if (!hits[i])
                    continue;
                const size_t p = chunk.start + i;
                std::atomic_ref<int>(diff[p]).fetch_add(1, std::memory_order_relaxed);
                std::atomic_ref<int>(diff[p + kmerSize]).fetch_sub(1, std::memory_order_relaxed);
            }
            return;
        }
        CKmerAPI KMCKmer(kmerSize);
        for (size_t i = 0; i < codes.size(); i++)
        {
            if (!valid[i])
                continue;
            KMCKmer.from_int(codes[i], kmerSize);
            if (KMCDatabase.IsKmer(KMCKmer))
            {
                const size_t p = chunk.start + i;
                std::atomic_ref<int>(diff[p]).fetch_add(1, std::memory_order_relaxed);
                std::atomic_ref<int>(diff[p + kmerSize]).fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    // Coverage-track writer shared by the CPU and GPU mapper paths:
    // prefix-sums the difference array and joins with commas.
    static string serializeCoverage(const vector<int> &diff, size_t seqSize)
    {
        string mappingStr;
        mappingStr.reserve(2 * seqSize);
        char buf[12];
        int coverage = 0;
        for (size_t i = 0; i < seqSize; i++)
        {
            coverage += diff[i];
            auto res = to_chars(buf, buf + sizeof(buf), coverage);
            mappingStr.append(buf, res.ptr);
            mappingStr.push_back(',');
        }
        if (!mappingStr.empty())
            mappingStr.pop_back();
        return mappingStr;
    }

    pair<string, string> getMappingFromSequence(const string &id, const string &sequence)
    {
        if (useLegacyPath())
            return getMappingFromSequenceLegacy(id, sequence);

        const size_t seqSize = sequence.size();

        vector<uint64_t> codes;
        vector<uint8_t> valid;
        KmerCodec(kmerSize).encodePositions(sequence, codes, valid);

        // Difference array + prefix pass: O(N) instead of k scattered
        // increments per hit.
        vector<int> diff(seqSize + 1, 0);
        CKmerAPI KMCKmer(kmerSize);
        for (size_t i = 0; i < codes.size(); i++)
        {
            if (!valid[i])
                continue;
            KMCKmer.from_int(codes[i], kmerSize);
            if (KMCDatabase.IsKmer(KMCKmer))
            {
                diff[i] += 1;
                diff[i + kmerSize] -= 1;
            }
        }

        return {id, serializeCoverage(diff, seqSize)};
    }

    pair<string,string> getMappingFromSequenceLegacy(const string &id, string sequence)
    {
        auto seqSize = sequence.size();
        vector<short> mapping(seqSize, 0);
        // guard: seqSize - kmerSize underflows for contigs shorter than k
        // (the original code crashed on them); coverage is all zeros
        if (seqSize < kmerSize)
        {
            string zeros;
            for (size_t i = 0; i < seqSize; i++)
                zeros += i + 1 < seqSize ? "0," : "0";
            return {id, zeros};
        }
        for (size_t i = 0; i <= seqSize - kmerSize; i++)
        {
            string kmer = sequence.substr(i, kmerSize);
            transform(kmer.begin(), kmer.end(), kmer.begin(), ::toupper);
            if (kmer.find_first_not_of("ACGT") != string::npos)
                continue;
            string revComplement = reverseComplement(kmer);
            string canonicalKmer = kmer > revComplement ? revComplement : kmer;
            CKmerAPI KMCKmer;
            KMCKmer.from_string(canonicalKmer);            
            if (KMCDatabase.IsKmer(KMCKmer))
            {
                for (size_t j = 0; j < kmerSize; j++)
                {
                    mapping[i + j] += 1;
                }
            }
        }

        string mappingStr;
        for (auto m : mapping)
        {
            mappingStr += to_string(m) + ",";
        }

        if (!mappingStr.empty())
        {
            mappingStr.pop_back();
        }

        return {id, mappingStr};
    }

    size_t getTotalLength(const vector<string> &sequences)
    {
        size_t totalLength = 0;
        for (const auto &seq : sequences)
        {
            totalLength += seq.size();
        }
        return totalLength;
    }

    size_t estimateNumChunks(const vector<string> &sequences, size_t chunkSize)
    {
        size_t totalLength = getTotalLength(sequences);
        return totalLength / chunkSize + sequences.size() * 2;
    }

    // A reference read and windowed: the contigs and the windows viewing
    // into them (the windows hold pointers into `parsed`, so the struct is
    // filled in place and never copied). Shared by the single-database
    // scan and panel mode, which reads and windows once for N databases.
    struct ReferenceWindows
    {
        ParseResult parsed;
        vector<SeqWindow> chunks;
        double readSec = 0, windowSec = 0;
        ReferenceWindows() = default;
        ReferenceWindows(const ReferenceWindows &) = delete;
        ReferenceWindows &operator=(const ReferenceWindows &) = delete;
    };

    void readReferenceWindows(const string &refPath, int windowSize, ReferenceWindows &out)
    {
        auto t0 = chrono::steady_clock::now();
        cout << "Reading sequences" << endl;
        bool compressed = refPath.find(".gz") != string::npos;
        out.parsed = readSequenceFile(refPath, compressed);
        auto t1 = chrono::steady_clock::now();
        out.readSec = chrono::duration<double>(t1 - t0).count();
        // References only: windows view into these strings, which live as
        // long as `out`.
        const auto &sequences = out.parsed.sequences;
        const auto &ids = out.parsed.ids;
        out.chunks.clear();
        out.chunks.reserve(estimateNumChunks(sequences, windowSize));
        chunkSequences(ids, sequences, windowSize, kmerSize, out.chunks);
        out.windowSec = chrono::duration<double>(chrono::steady_clock::now() - t1).count();
        cout << "Number of chunks: " << out.chunks.size() << endl;
    }

    // Output files are written as "<outPath>.partial" and renamed once
    // complete. A failed open, write or close (a read-only or full volume)
    // throws, so the caller counts the reference as failed and the exit
    // status says so, and no truncated file is left under the final name
    // for the next run's skip-if-exists rule to keep.
    template <typename WriteFn>
    static void writeOutputFile(const string &outPath, WriteFn write)
    {
        const string tmpPath = outPath + ".partial";
        std::error_code ec;
        {
            errno = 0;
            ofstream out(tmpPath);
            if (!out.is_open())
                throw std::runtime_error("Unable to open " + tmpPath + " for writing" +
                                         (errno ? string(": ") + strerror(errno) : string()));
            write(out);
            out.close();
            if (!out)
            {
                std::filesystem::remove(tmpPath, ec);
                throw std::runtime_error("Writing " + outPath + " failed (disk full or I/O error)");
            }
        }
        std::filesystem::rename(tmpPath, outPath, ec);
        if (ec)
        {
            std::error_code ignored;
            std::filesystem::remove(tmpPath, ignored);
            throw std::runtime_error("Unable to move " + tmpPath + " to " + outPath + ": " +
                                     ec.message());
        }
    }

    // The TSV writer, shared by every scan path (a panel's N files are N
    // calls of it).
    static void writeStats(const string &outPath, const vector<Window> &statsResult)
    {
        writeOutputFile(outPath, [&statsResult](ofstream &statsFile)
        {
            statsFile << "seqname\tstart\tend\ttotal_kmers\tobserved_kmers\tvariations\tkmer_distance\n";
            for (size_t i = 0; i < statsResult.size(); i++)
            {
                auto stats = statsResult[i];
                statsFile << stats.id << '\t' << stats.start << '\t' << stats.end << '\t'
                          << stats.totalKmers << '\t' << stats.observedKmers << '\t'
                          << stats.variations << '\t' << stats.kmerDistance << endl;
            }
        });
    }

    void processReference(string refPath, string outPath, int windowSize = 50000)
    {
        // Phase timers for the host-side breakdown (one "Scan phases" line
        // per reference, CPU and GPU paths alike).
        auto phaseClock = chrono::steady_clock::now();
        auto lap = [&phaseClock]()
        {
            const auto now = chrono::steady_clock::now();
            const double s = chrono::duration<double>(now - phaseClock).count();
            phaseClock = now;
            return s;
        };
        double setupSec = 0, readSec = 0, windowSec = 0, statsSec = 0, writeSec = 0;
#ifdef FASTIBS_CUDA
        // Lazy device-index upload: only the stats pipeline probes on the
        // GPU (until Phase 4), so the mapper and intersect tools never pay
        // for the upload. One upload serves all references (amortized).
        if (!gpuInitAttempted && !useLegacyPath() && gpuRequested())
        {
            gpuInitAttempted = true;
            initGpuIndex();
        }
#endif
        cout << "Window size: " << windowSize << endl;
        cout << "Kmer size: " << kmerSize << endl;
        string path = useLegacyPath() ? "legacy strings (CPU-BASE)" : "rolling codec (CPU-OPT)";
        // CPU-BATCHED: the CPU path with the K-way pipelined probe; a GPU
        // run (its own probe) and the legacy oracle leave it unused.
        bool batched = cpuBatchedApplies();
        bool streamed = false;
#ifdef FASTIBS_CUDA
        if (gpuActive())
            batched = false;
        if (!gpuIndexes.empty() && gpuSyncRequested())
            path = "rolling codec + GPU per-window probe (GPU-BS sync)";
        else if (gpuActive())
        {
            streamed = true;
            const string backend = !gpuHashIndexes.empty() ? "GPU-HASH" : "GPU-BS";
            // The engine sorts on one device only (it says so when it
            // drops --sort-queries), so the label follows what runs.
            const bool sorted = gpuOptions().sortQueries &&
                                max(gpuIndexes.size(), gpuHashIndexes.size()) == 1;
            const string order = sorted ? "-SORTED" : "";
            path = "fused GPU encode+probe, streamed (" + backend + order + ")";
        }
#endif
        if (batched)
            path = "rolling codec + " + to_string(cpuBatchK()) +
                   "-way pipelined probe of the host index image (CPU-BATCHED)";
        cout << "K-mer path: " << path << endl;

        // CPU probing (also the legacy oracle) needs the host-RAM view;
        // open it before the pool starts probing concurrently. GPU-probed
        // runs skip this entirely — the device index is read straight from
        // the on-disk files; the batched CPU probe reads the host image
        // of the same files instead of the KMC view.
        bool needRa = true;
#ifdef FASTIBS_CUDA
        needRa = useLegacyPath() || !gpuActive();
#endif
        if (batched)
            ensureHostIndex();
        else if (needRa)
            ensureRaOpen();
        setupSec = lap(); // device upload (GPU, first reference), host image or RA open (CPU)

        ReferenceWindows ref;
        readReferenceWindows(refPath, windowSize, ref);
        readSec = ref.readSec;
        windowSec = ref.windowSec;
        lap();
        const auto &chunks = ref.chunks;

        /************************************************************/

        cout << "Calculating stats" << endl;
        vector<Window> statsResult(chunks.size());
        thread_pool pool(poolThreads());
        boost::progress_display progressBar(chunks.size());
#ifdef FASTIBS_CUDA
        if (streamed)
        {
            processReferenceGpu(chunks, statsResult, pool, progressBar);
        }
        else
#endif
        {
            static std::mutex progressMx; // progress_display is not thread-safe
            for (size_t i = 0; i < chunks.size(); i++)
            {
                pool.push_task([i, this, &chunks, &statsResult, &progressBar]
                               {
                                statsResult[i] = getStatsFromSequence(chunks[i]);
                                std::lock_guard<std::mutex> lk(progressMx);
                                ++progressBar; });
            }
            pool.wait_for_tasks();
        }
        cout << endl;
        statsSec = lap();

        /************************************************************/

        cout << "Writing stats to file" << endl;
        writeStats(outPath, statsResult);
        writeSec = lap();
        // Where a scan's wall time goes: setup is the device upload (GPU
        // path, first reference only) or the random-access open (CPU path);
        // stats is the probe+statistics block, on the GPU path the streamed
        // pipeline's wall time (its own line says how much of it the host
        // spent waiting for the device).
        cout << "Scan phases: setup " << setupSec << " s, read " << readSec
             << " s, window " << windowSec << " s, stats " << statsSec
             << " s, write " << writeSec << " s" << endl;
    }

void produceMapping(string refPath, string outPath)
    {
#ifdef FASTIBS_CUDA
        if (!gpuInitAttempted && !useLegacyPath() && gpuRequested())
        {
            gpuInitAttempted = true;
            initGpuIndex();
        }
#endif
        cout << "Kmer size: " << kmerSize << endl;
        string path = useLegacyPath() ? "legacy strings (CPU-BASE)" : "rolling codec (CPU-OPT)";
        bool gpuMapping = false;
#ifdef FASTIBS_CUDA
        if (gpuActive() && !useLegacyPath())
        {
            gpuMapping = true;
            path = string("fused GPU encode+probe, streamed (") +
                   (!gpuHashIndexes.empty() ? "GPU-HASH" : "GPU-BS") +
                   (gpuOptions().sortQueries &&
                            max(gpuIndexes.size(), gpuHashIndexes.size()) == 1
                        ? "-SORTED"
                        : "") +
                   ")";
        }
#endif
        // CPU-BATCHED: the chunk-parallel CPU mapper with the K-way
        // pipelined probe (accumulateMappingChunk takes the batched branch
        // once the host image is loaded).
        const bool batched = !gpuMapping && cpuBatchedApplies();
        if (batched)
            path = "rolling codec + " + to_string(cpuBatchK()) +
                   "-way pipelined probe of the host index image (CPU-BATCHED)";
        cout << "K-mer path: " << path << endl;
        if (batched)
            ensureHostIndex(); // the batched probe reads the host image
        else if (!gpuMapping)
            ensureRaOpen(); // CPU probing needs the host-RAM view

        cout << "Reading sequences" << endl;
        bool compressed = refPath.find(".gz") != string::npos;
        auto result = readSequenceFile(refPath, compressed);
        const auto &sequences = result.sequences;
        const auto &ids = result.ids;

        /************************************************************/

        cout << "Calculating mapping" << endl;
        vector<pair<string, string>> mappingResult(sequences.size());
        thread_pool pool(poolThreads());
        boost::progress_display progressBar(sequences.size());
        // Phase timers: the coverage track is serialized and written
        // serially per contig on both the CPU and GPU paths, so knowing
        // the probe/serialize split is what tells us where the mapper's
        // remaining time goes.
        double probeSec = 0, serializeSec = 0;
        auto phaseStart = chrono::high_resolution_clock::now();
        auto lap = [&]
        {
            const auto now = chrono::high_resolution_clock::now();
            const double d = chrono::duration<double>(now - phaseStart).count();
            phaseStart = now;
            return d;
        };

        if (useLegacyPath())
        {
            // CPU-BASE: one task per contig, as the original code did
            // (probe and serialize are interleaved and not separable).
            static std::mutex legacyProgressMx; // progress_display is not thread-safe
            for (size_t i = 0; i < sequences.size(); i++)
            {
                pool.push_task([i, this, &sequences, &ids, &mappingResult, &progressBar]
                               {
                                mappingResult[i] = getMappingFromSequence(ids[i], sequences[i]);
                                std::lock_guard<std::mutex> lk(legacyProgressMx);
                                ++progressBar; });
            }
            pool.wait_for_tasks();
            probeSec = lap();
        }
        else
        {
            // Both accelerated paths accumulate hits into per-contig
            // difference arrays, then share one serialization phase.
            vector<vector<int>> diffs(sequences.size());
            for (size_t c = 0; c < sequences.size(); c++)
                diffs[c].assign(sequences[c].size() + 1, 0);

#ifdef FASTIBS_CUDA
            if (gpuMapping)
            {
                produceMappingGpu(ids, sequences, diffs, pool);
            }
            else
#endif
            {
                // CPU-OPT: chunk-parallel. One task per contig leaves all
                // but one core idle whenever a reference is dominated by a
                // single large contig (the common case for plant
                // genomes), which would make this baseline effectively
                // serial. The contig is split with the same
                // position-partitioning the GPU path uses, so every
                // k-mer position is probed exactly once.
                vector<SeqWindow> chunks;
                vector<size_t> chunkContig;
                chunkForMapping(ids, sequences, max<size_t>(chunkSize, 2 * kmerSize),
                                kmerSize, chunks, chunkContig);
                for (size_t j = 0; j < chunks.size(); j++)
                {
                    pool.push_task([j, this, &chunks, &chunkContig, &diffs]
                                   { accumulateMappingChunk(chunks[j], diffs[chunkContig[j]]); });
                }
                pool.wait_for_tasks();
            }
            probeSec = lap();

            static std::mutex progressMx;
            for (size_t c = 0; c < sequences.size(); c++)
            {
                pool.push_task([c, &ids, &sequences, &diffs, &mappingResult, &progressBar]
                               {
                                   mappingResult[c] = {ids[c], serializeCoverage(diffs[c], sequences[c].size())};
                                   std::lock_guard<std::mutex> lk(progressMx);
                                   ++progressBar;
                               });
            }
            pool.wait_for_tasks();
            serializeSec = lap();
        }
        cout << endl;

        /************************************************************/

        cout << "Writing mapping to file" << endl;
        writeOutputFile(outPath, [&mappingResult](ofstream &mappingFile)
        {
            for (size_t i = 0; i < mappingResult.size(); i++)
            {
                auto [id, seq] = mappingResult[i];
                mappingFile << id << '\n' << seq << endl;
            }
        });
        // The coverage track is one text line per contig — gigabytes for a
        // multi-Gbp reference — and this write is common to every path, so
        // it bounds any mapper speedup regardless of how fast probing is.
        cout << "Mapping phases: probe " << probeSec << " s, serialize "
             << serializeSec << " s, write " << lap() << " s" << endl;
    }


private:
#ifdef FASTIBS_CUDA
    // Generic streamed pipeline (Phases 2, 4 and 5): windows are grouped
    // into batches of ~batchMB of raw sequence; each batch is 2-bit packed
    // into a slot's pinned staging buffer by the thread pool, moved H2D on
    // the slot's stream, processed by the fused encode+probe kernel, and
    // its bit-packed hit/validity masks moved D2H — while other slots'
    // copies and kernels overlap. With the index split across devices, a
    // batch goes to ONE device (round-robin), which encodes it and routes
    // each query to the device holding its shard over NVLink / PCIe
    // peer-to-peer; the owners' hit flags come back to it and it returns
    // the assembled hit mask — so the host side is the single-GPU
    // pipeline: one hit mask and one validity mask per batch, no merge.
    // The per-window result pass runs on the same thread pool, overlapping
    // everything: consume(winIdx, hitWords, validWords, desc) is called
    // from a pool task once the window's batch completes and must be
    // thread-safe.
    template <typename ConsumeFn>
    void runGpuPipeline(const vector<SeqWindow> &chunks, thread_pool &pool,
                        ConsumeFn consume)
    {
        vector<vector<const DeviceKmerIndex *>> bsPerDb;
        if (!gpuIndexes.empty())
        {
            bsPerDb.emplace_back();
            for (const auto &p : gpuIndexes)
                bsPerDb.back().push_back(p.get());
        }
        vector<const DeviceHashIndex *> hashShards;
        for (const auto &p : gpuHashIndexes)
            hashShards.push_back(p.get());
        runEngineLoop(bsPerDb, hashShards, kmerSize, chunks, pool,
                      [&consume](size_t winIdx, const EncodeProbeEngine::HostSlot &hs,
                                 const fastibs::WindowDesc &d)
                      { consume(winIdx, hs.hitWords, hs.validWords, d); });
    }

    // The pipeline driver itself, over any set of resident images:
    // bsPerDb[i][d] is shard d of database i (one database: the scan above;
    // several: panel mode, docs/MULTIDB_SCAN.md, where every batch is
    // encoded once and probed against all of them), or the hash shards.
    // consume(winIdx, slot, desc) reads the slot's masks: slot.hits(i) is
    // database i's, slot.validWords the shared validity mask.
    template <typename ConsumeFn>
    static void runEngineLoop(const vector<vector<const DeviceKmerIndex *>> &bsPerDb,
                              const vector<const DeviceHashIndex *> &hashShards,
                              uint32_t k, const vector<SeqWindow> &chunks,
                              thread_pool &pool, ConsumeFn consume)
    {
        using fastibs::WindowDesc;
        using Batch = fastibs::WindowBatch; // winIdx = global chunk indices

        const int nDbs = (int)max<size_t>(1, bsPerDb.size());
        // The engine must be able to hold the largest single window in one
        // batch (a window is never split), so the configured batch size is
        // a floor, not a ceiling, when windows are huge.
        size_t maxWinBases = 0;
        for (const auto &c : chunks)
            maxWinBases = max(maxWinBases, c.seq.size());
        size_t batchBases =
            max((size_t)gpuOptions().batchMB << 20, maxWinBases);
        const int depth = max(1, gpuOptions().streams);
        // Accounting for the breakdown: time the submitting thread spent
        // blocked on the device (engine.wait) and on the pool's packing,
        // and the pool's thread-seconds in packing and consuming.
        double waitSec = 0, packWaitSec = 0;
        size_t nBatches = 0;
        atomic<uint64_t> packNs{0}, consumeNs{0};
        auto windowCapFor = [](size_t bases)
        { return max<size_t>(4096, bases / 256); };

        const vector<const DeviceKmerIndex *> bsShards =
            bsPerDb.empty() ? vector<const DeviceKmerIndex *>{} : bsPerDb[0];

        // The pipeline's buffers live next to the shard already resident on
        // each device, and with several devices the routed path needs ~21
        // bytes per batch position per slot: 128 MB batches x 3 streams are
        // 9.85 GB per device, more than a 16 GB part has left beside an
        // 8 GB shard (job 68747: "partition codes alloc: out of memory").
        // Halve the batch until the estimate fits the least free memory of
        // the devices used, down to the largest window (a window is never
        // split); say so, since the batch size is a knob of the paper.
        {
            vector<int> devices;
            for (const auto *p : bsShards)
                devices.push_back(p->device());
            for (const auto *p : hashShards)
                devices.push_back(p->device());
            size_t freeMin = 0;
            bool known = false;
            for (int dev : devices)
                if (const size_t f = DeviceKmerIndex::freeMemory(dev))
                {
                    freeMin = known ? min(freeMin, f) : f;
                    known = true;
                }
            if (known)
            {
                const int n = (int)devices.size();
                const bool direct = n > 1 && EncodeProbeEngine::peerAccessible(devices);
                const size_t margin = 256ull << 20; // context, kernels, fragmentation
                auto need = [&](size_t bases)
                {
                    return EncodeProbeEngine::estimateDeviceBytes(
                        n, direct, bases, windowCapFor(bases), depth,
                        gpuOptions().sortQueries, nDbs);
                };
                const size_t requested = batchBases;
                while (need(batchBases) + margin > freeMin &&
                       batchBases / 2 >= max<size_t>(maxWinBases, 1ull << 20))
                    batchBases /= 2;
                char msg[240];
                if (batchBases != requested)
                {
                    snprintf(msg, sizeof msg,
                             "[gpu] pipeline buffers: %zu MB batches x %d streams need %.2f GB per device, "
                             "%.2f GB free beside the shard; using %zu MB batches (%.2f GB)",
                             requested >> 20, depth, need(requested) / 1e9, freeMin / 1e9,
                             batchBases >> 20, need(batchBases) / 1e9);
                    cout << msg << endl;
                }
                else if (need(batchBases) + margin > freeMin)
                {
                    snprintf(msg, sizeof msg,
                             "[gpu] pipeline buffers (%.2f GB per device at the smallest batch that "
                             "holds the largest window) exceed the %.2f GB free beside the shard; "
                             "reduce --streams or use more devices",
                             need(batchBases) / 1e9, freeMin / 1e9);
                    cout << msg << endl;
                }
            }
        }
        const size_t windowCap = windowCapFor(batchBases);
        // Several databases: the panel constructor (ported index, fused
        // ordering); otherwise the engine exactly as before.
        std::unique_ptr<EncodeProbeEngine> enginePtr;
        if (nDbs > 1)
            enginePtr = std::make_unique<EncodeProbeEngine>(
                bsPerDb, k, batchBases, windowCap, depth, gpuOptions().pinLut);
        else
            enginePtr = std::make_unique<EncodeProbeEngine>(
                bsShards, hashShards, k, batchBases, windowCap, depth,
                gpuOptions().sortQueries, gpuOptions().pinLut);
        EncodeProbeEngine &engine = *enginePtr;
        const int nDev = engine.shards();
        {
            char buf[96];
            snprintf(buf, sizeof buf, ", %.2f GB of pipeline buffers per device",
                     engine.deviceBufferBytes() / 1e9);
            cout << "[gpu] pipeline: " << (batchBases >> 20) << " MB batches, "
                 << depth << " streams"
                 << (nDbs > 1 ? ", " + to_string(nDbs) + " databases probed per encode" : "")
                 << (nDev > 1 ? ", index split across " + to_string(nDev) +
                                    " devices (one encoder per batch, queries routed " +
                                    (engine.peerDirect() ? "in place over NVLink/P2P)"
                                                         : "via host-staged copies)")
                              : "")
                 << (gpuOptions().sortQueries && nDev == 1 ? ", sorted queries" : "")
                 << (gpuOptions().pinLut ? ", LUT pinned" : "") << buf << endl;
        }

        vector<shared_ptr<Batch>> slotBatch(depth);
        // Completed-batch consumer tasks read the slot's pinned result
        // buffers; the counter gates the next submit on that slot (whose
        // D2H would overwrite them).
        vector<shared_ptr<atomic<size_t>>> slotPending(depth);
        for (auto &p : slotPending)
            p = make_shared<atomic<size_t>>(0);

        auto dispatchConsume = [&](int s)
        {
            shared_ptr<Batch> b = std::move(slotBatch[s]);
            if (!b || b->winIdx.empty())
                return;
            const auto hs = engine.host(s);
            auto pending = slotPending[s];
            pending->store(b->winIdx.size(), std::memory_order_release);
            for (size_t j = 0; j < b->winIdx.size(); j++)
            {
                pool.push_task([b, hs, j, consume, pending, &consumeNs]
                {
                    const WindowDesc &d = b->layout.descs[j];
                    const auto t0 = chrono::steady_clock::now();
                    consume(b->winIdx[j], hs, d);
                    consumeNs.fetch_add((uint64_t)chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
                    pending->fetch_sub(1, std::memory_order_release);
                });
            }
        };

        auto flushBatch = [&](int s, shared_ptr<Batch> b)
        {
            // Reclaim the slot: finish its in-flight batch, hand its
            // result pass to the pool, then pack the new batch into the
            // slot's staging buffers (packing writes seq/inv, consumers
            // read hit/valid — disjoint, so they overlap).
            {
                const auto tw = chrono::steady_clock::now();
                engine.wait(s);
                waitSec += chrono::duration<double>(chrono::steady_clock::now() - tw).count();
            }
            dispatchConsume(s);

            const auto hs = engine.host(s);
            atomic<size_t> packPending{b->winIdx.size()};
            for (size_t j = 0; j < b->winIdx.size(); j++)
            {
                pool.push_task([&, j]
                {
                    const WindowDesc &d = b->layout.descs[j];
                    const auto t0 = chrono::steady_clock::now();
                    fastibs::packWindow(chunks[b->winIdx[j]].seq,
                                        hs.seqWords + d.seqWordOff,
                                        hs.invWords + d.invWordOff);
                    hs.descs[j] = d;
                    packNs.fetch_add((uint64_t)chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
                    packPending.fetch_sub(1, std::memory_order_release);
                });
            }
            {
                const auto tw = chrono::steady_clock::now();
                while (packPending.load(std::memory_order_acquire))
                    std::this_thread::yield();
                packWaitSec += chrono::duration<double>(chrono::steady_clock::now() - tw).count();
            }
            // The slot's previous consumers must be done before D2H
            // overwrites the result buffers.
            while (slotPending[s]->load(std::memory_order_acquire))
                std::this_thread::yield();

            engine.submit(s, (uint32_t)b->winIdx.size(), b->layout.seqWords,
                          b->layout.invWords, b->layout.outWords);
            nBatches++;
            slotBatch[s] = std::move(b);
        };

        // On any error (a CUDA failure surfaces in engine.wait/submit),
        // already-dispatched pack/consume tasks must finish before the
        // exception unwinds this frame — they read the engine's pinned
        // buffers, which unwinding would destroy under them.
        int slot = 0;
        try
        {
            // Windows are never split: buildBatches guarantees every batch
            // fits the slot capacities (the engine was sized to the
            // largest window above) and never emits an empty batch.
            fastibs::buildBatches(
                chunks.size(), k, engine.caps(),
                [&](size_t i) { return chunks[i].seq.size(); },
                [&](Batch &&b)
                {
                    flushBatch(slot, make_shared<Batch>(std::move(b)));
                    slot = (slot + 1) % depth;
                });
            // Drain the ring.
            for (int d = 0; d < depth; d++)
            {
                const int s = (slot + d) % depth;
                const auto tw = chrono::steady_clock::now();
                engine.wait(s);
                waitSec += chrono::duration<double>(chrono::steady_clock::now() - tw).count();
                dispatchConsume(s);
            }
            pool.wait_for_tasks();
            // If waitSec is small against the pipeline's wall time, the host
            // never waited for the device: the scan is host-bound.
            cout << "[gpu] pipeline: " << nBatches << " batches; the submitting thread waited "
                 << waitSec << " s for the device and " << packWaitSec
                 << " s for packing; pool work: pack " << packNs.load() / 1e9
                 << " thread-s, consume " << consumeNs.load() / 1e9 << " thread-s on "
                 << pool.get_thread_count() << " threads" << endl;
        }
        catch (...)
        {
            pool.wait_for_tasks();
            throw;
        }
    }

    // Phase 2 consumer: run-length statistics per window.
    void processReferenceGpu(const vector<SeqWindow> &chunks,
                             vector<Window> &statsResult, thread_pool &pool,
                             boost::progress_display &progressBar)
    {
        const uint32_t k = kmerSize;
        // Every window's identity is known up front; windows shorter than
        // k produce no queries and never visit the device.
        for (size_t i = 0; i < chunks.size(); i++)
        {
            statsResult[i].id = *chunks[i].id;
            statsResult[i].start = chunks[i].start;
            statsResult[i].end = chunks[i].end;
            if (chunks[i].seq.size() < k)
                ++progressBar;
        }

        static std::mutex progressMx; // boost::progress_display is not thread-safe
        runGpuPipeline(chunks, pool,
                       [&statsResult, &progressBar, k](size_t winIdx,
                                                       const uint32_t *hitWords,
                                                       const uint32_t *validWords,
                                                       const fastibs::WindowDesc &d)
        {
            Window &w = statsResult[winIdx];
            fastibs::BitStats bs;
            fastibs::BitStats::run(hitWords, validWords, d.outWordOff, d.nPos,
                                   bs, [&](uint64_t gap)
                                   { w.addVariationOriginal((int)gap, (int)k); });
            w.totalKmers = (int)bs.totalKmers;
            w.observedKmers = (int)bs.observedKmers;
            std::lock_guard<std::mutex> lk(progressMx);
            ++progressBar;
        });
    }

    // Phase 4 consumer: per-base coverage mapping. Contigs are chunked
    // with a k-1 base overlap and a step of (chunk - k + 1) POSITIONS, so
    // unlike chunkSequences' windows (which share their boundary position
    // by design) every k-mer position of a contig appears in exactly one
    // chunk. Each hit at contig position p adds +1/-1 to the contig's
    // difference array at p and p+k — the O(N) replacement for k scattered
    // increments — and one prefix pass per contig materializes the track.
    // Consumers of adjacent chunks touch overlapping difference entries
    // (the -1 of one chunk lands in the next chunk's range), so the adds
    // are atomic.
    // Accumulates coverage hits into per-contig difference arrays (sized
    // and zeroed by the caller, which also owns the serialization phase).
    void produceMappingGpu(const vector<string> &ids,
                           const vector<string> &sequences,
                           vector<vector<int>> &diffs,
                           thread_pool &pool)
    {
        const uint32_t k = kmerSize;
        const size_t W = max<size_t>(chunkSize, (size_t)2 * k);

        vector<SeqWindow> chunks;    // start = contig-global base offset
        vector<size_t> chunkContig;  // per chunk: contig index
        // Contigs shorter than k have no positions: their coverage stays
        // all zeros (matching the CPU path); no chunks are generated.
        chunkForMapping(ids, sequences, W, k, chunks, chunkContig);

        runGpuPipeline(chunks, pool,
                       [&chunks, &chunkContig, &diffs, k](size_t winIdx,
                                                          const uint32_t *hitWords,
                                                          const uint32_t *validWords,
                                                          const fastibs::WindowDesc &d)
        {
            vector<int> &diff = diffs[chunkContig[winIdx]];
            const size_t base = chunks[winIdx].start;
            for (uint32_t p = 0; p < d.nPos; p++)
            {
                const uint64_t word = d.outWordOff + p / 32;
                const uint32_t bit = p & 31;
                if (((validWords[word] >> bit) & (hitWords[word] >> bit)) & 1)
                {
                    std::atomic_ref<int>(diff[base + p]).fetch_add(
                        1, std::memory_order_relaxed);
                    std::atomic_ref<int>(diff[base + p + k]).fetch_sub(
                        1, std::memory_order_relaxed);
                }
            }
        });
        // The prefix pass and serialization are the caller's phase, so
        // both accelerated paths share one implementation and one timer.
    }
#endif

    uint kmerSize, chunkSize = CHUNK_SIZE;
    string sourcePath;
    CKMCFile KMCDatabase;
    CKMCFileInfo KMCInfo;
    bool raOpened = false;
    std::mutex raOpenMutex;
    // Host image of the index, loaded only for the batched CPU probe
    // (ensureHostIndex); non-null is what selects that probe per window.
    std::unique_ptr<KmcRaIndex> hostIndex;
#ifdef FASTIBS_CUDA
    std::vector<std::unique_ptr<DeviceKmerIndex>> gpuIndexes;
    std::vector<std::unique_ptr<DeviceHashIndex>> gpuHashIndexes;
    bool gpuInitAttempted = false;
#endif
};
