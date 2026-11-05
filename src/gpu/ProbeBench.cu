// Probe microbenchmark: sustained membership-probe throughput of the
// device-resident structures vs a GUPS-style random-access ceiling.
// Phase 1: the ported KMC index (GPU-BS). Phase 3 ablations: value-sorted
// queries (--sorted, coalesced searches), the hash-set backend (--hash),
// and an L2-persisting prefix LUT (--pin-lut). Upload/build time and
// footprints are logged by the index classes.
//
// Usage: probe_bench <kmc_db_path_prefix> [num_queries=16000000] [reps=10]
//                    [--sorted] [--hash] [--hash-lf=F] [--pin-lut]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "DeviceHashIndex.cuh"
#include "DeviceKmerIndex.cuh"
#include "HashCore.hpp"
#include "KmcRaIndex.hpp"

// Hash-backend probe: one query per thread against the open-addressing
// table (fastibs::hashContains — the same shared function the fused
// pipeline kernel uses).
__global__ void hashProbeKernel(const uint64_t *__restrict__ queries, size_t n,
                                const uint64_t *__restrict__ table,
                                fastibs::HashParams P,
                                uint8_t *__restrict__ hits)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        hits[i] = fastibs::hashContains(__ldg(queries + i), table, P) ? 1 : 0;
}

#define CUDA_CHECK(x)                                                     \
    do                                                                    \
    {                                                                     \
        cudaError_t err__ = (x);                                          \
        if (err__ != cudaSuccess)                                         \
        {                                                                 \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(err__), \
                   __LINE__);                                             \
            return 1;                                                     \
        }                                                                 \
    } while (0)

// GUPS-style ceiling: each thread performs `loadsPerThread` independent
// pseudo-random 8-byte loads over a working set of `words` uint64s — the
// same access pattern class as the probe's binary-search steps, minus the
// data dependence. Reported loads/s bound what any lookup structure of this
// footprint can sustain.
__global__ void gupsKernel(const uint64_t *__restrict__ data, size_t words,
                           int loadsPerThread, uint64_t *__restrict__ sink)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t x = tid * 0x9E3779B97F4A7C15ull + 1;
    uint64_t acc = 0;
    for (int i = 0; i < loadsPerThread; ++i)
    {
        // xorshift64* address generation: independent (not chained) loads
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        acc += __ldg(data + (x * 0x2545F4914F6CDD1Dull) % words);
    }
    if (acc == 0xdeadbeef) // never true; prevents dead-code elimination
        *sink = acc;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: %s <kmc_db_path_prefix> [num_queries] [reps]\n", argv[0]);
        return 1;
    }
    const std::string db = argv[1];
    size_t Q = 16000000ull;
    int reps = 10;
    bool doSorted = false, doHash = false, pinLut = false;
    double hashLf = 0.6;
    int positional = 0;
    for (int a = 2; a < argc; a++)
    {
        const std::string arg = argv[a];
        if (arg == "--sorted")
            doSorted = true;
        else if (arg == "--hash")
            doHash = true;
        else if (arg.rfind("--hash-lf=", 0) == 0)
        {
            doHash = true;
            hashLf = atof(arg.c_str() + 10);
            if (!(hashLf >= 0.1 && hashLf <= 0.95)) // negated form rejects NaN
            {
                printf("invalid %s (expected 0.1 .. 0.95)\n", arg.c_str());
                return 1;
            }
        }
        else if (arg == "--pin-lut")
            pinLut = true;
        else if (positional == 0)
        {
            Q = strtoull(arg.c_str(), nullptr, 10);
            positional++;
        }
        else if (positional == 1)
        {
            reps = atoi(arg.c_str());
            positional++;
        }
        else
        {
            printf("unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (DeviceKmerIndex::deviceCount() == 0)
    {
        printf("no CUDA device available (%s)\n", DeviceKmerIndex::deviceCountError());
        return 1;
    }

    printf("loading %s\n", db.c_str());
    KmcRaIndex host(db);
    const auto &P = host.getParams();
    printf("k=%u p=%u sig_len=%u total_kmers=%llu\n", P.kmerLength,
           P.lutPrefixLength, P.signatureLen,
           (unsigned long long)P.totalKmers);

    // unique_ptr so the BS index can be released before the hash build:
    // large hash tables and the BS index need not coexist on the device.
    auto indexPtr = std::make_unique<DeviceKmerIndex>(host);
    DeviceKmerIndex &index = *indexPtr;

    // Uniform random queries — adjacent reference positions produce
    // unrelated k-mer values, so this matches the production access
    // pattern; the measured hit rate of a sample is reported for context
    // (hit and miss probes cost the same order of loads).
    std::mt19937_64 rng(20260830);
    const uint64_t kmerMask =
        P.kmerLength == 32 ? ~0ull : (1ull << (2 * P.kmerLength)) - 1;
    std::vector<uint64_t> queries(Q);
    for (auto &q : queries)
        q = rng() & kmerMask;
    size_t hits = 0;
    for (size_t i = 0; i < std::min<size_t>(Q, 100000); ++i)
        hits += host.probe(queries[i]);
    printf("query sample hit rate: %.2f%%\n",
           100.0 * hits / std::min<size_t>(Q, 100000));

    uint64_t *d_queries = nullptr;
    uint8_t *d_hits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_queries, Q * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_hits, Q));
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), Q * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (pinLut)
        index.setLutPersisting(stream);

    // warmup + timed probe reps (GPU-BS baseline)
    index.probe(d_queries, nullptr, Q, d_hits, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r)
        index.probe(d_queries, nullptr, Q, d_hits, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    const double probeSec =
        std::chrono::duration<double>(t1 - t0).count() / reps;
    const double gqps = Q / probeSec / 1e9;
    printf("BS%s      : %.3f ms / %zu queries -> %.3f Gq/s\n",
           pinLut ? "+pinned-LUT" : "           ", probeSec * 1e3, Q, gqps);

    if (doSorted)
    {
        // Value-sorted queries: measures the coalescing gain of the sort
        // ablation in isolation (sort time reported separately; the
        // end-to-end cost including position payload + hit scatter is
        // measured by fastibs --sort-queries).
        uint64_t *d_sorted = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sorted, Q * sizeof(uint64_t)));
        void *d_tmp = nullptr;
        size_t tmpBytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, tmpBytes, d_queries,
                                                  d_sorted, Q, 0,
                                                  2 * (int)P.kmerLength, stream));
        CUDA_CHECK(cudaMalloc(&d_tmp, tmpBytes));
        // Warm up, then time `reps` sorts, exactly as the probes are timed.
        // A single cold call also pays CUDA's lazy module load for the CUB
        // kernels: it read 1.8 ms in one job and 6.1 ms in another on the
        // same node with identical probe times (jobs 55617 / 59626).
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_tmp, tmpBytes, d_queries,
                                                  d_sorted, Q, 0,
                                                  2 * (int)P.kmerLength, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r)
        {
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_tmp, tmpBytes, d_queries,
                                                      d_sorted, Q, 0,
                                                      2 * (int)P.kmerLength, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t1 = std::chrono::high_resolution_clock::now();
        const double sortSec =
            std::chrono::duration<double>(t1 - t0).count() / reps;

        index.probe(d_sorted, nullptr, Q, d_hits, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r)
            index.probe(d_sorted, nullptr, Q, d_hits, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        t1 = std::chrono::high_resolution_clock::now();
        const double sortedSec =
            std::chrono::duration<double>(t1 - t0).count() / reps;
        printf("BS-SORTED       : %.3f ms / %zu queries -> %.3f Gq/s "
               "(+ %.3f ms sort, %.2fx probe speedup)\n",
               sortedSec * 1e3, Q, Q / sortedSec / 1e9, sortSec * 1e3,
               probeSec / sortedSec);
        cudaFree(d_sorted);
        cudaFree(d_tmp);
    }

    if (doHash)
    {
        indexPtr.reset(); // release the BS index before the table build
        try
        {
            DeviceHashIndex hashIndex(host, hashLf);
            const auto hv = hashIndex.view();
            const int block = 256;
            const int grid =
                (int)std::min<size_t>((Q + block - 1) / block, 65535);
            hashProbeKernel<<<grid, block, 0, stream>>>(d_queries, Q, hv.table,
                                                        hv.params, d_hits);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < reps; ++r)
                hashProbeKernel<<<grid, block, 0, stream>>>(
                    d_queries, Q, hv.table, hv.params, d_hits);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            t1 = std::chrono::high_resolution_clock::now();
            const double hashSec =
                std::chrono::duration<double>(t1 - t0).count() / reps;
            printf("HASH (lf=%.2f)  : %.3f ms / %zu queries -> %.3f Gq/s "
                   "(%.2fx vs BS)\n",
                   hashLf, hashSec * 1e3, Q, Q / hashSec / 1e9,
                   probeSec / hashSec);
        }
        catch (const std::exception &e)
        {
            printf("HASH backend failed: %s\n", e.what());
        }
    }

    // GUPS ceiling over the suffix-array working set
    const size_t words =
        std::max<size_t>(P.totalKmers * (host.recordsAre32Bit() ? 4 : 8) / 8,
                         1);
    uint64_t *d_gups = nullptr;
    uint64_t *d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gups, words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_gups, 0x5a, words * sizeof(uint64_t)));
    const int block = 256, grid = 2048, loadsPerThread = 64;
    gupsKernel<<<grid, block>>>(d_gups, words, loadsPerThread, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r)
        gupsKernel<<<grid, block>>>(d_gups, words, loadsPerThread, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    const double gupsSec =
        std::chrono::duration<double>(t1 - t0).count() / reps;
    const double gLoads =
        (double)grid * block * loadsPerThread / gupsSec / 1e9;
    printf("GUPS ceiling: %.3f Gload/s over %.2f MB working set\n", gLoads,
           words * 8.0 / 1048576.0);

    // A probe costs ~log2(bucket) suffix loads + LUT/signature loads; report
    // the implied loads/s so the fraction-of-ceiling is directly visible.
    const double bucketLoads =
        P.totalKmers > 0
            ? (double)(64 - __builtin_clzll(
                                std::max<uint64_t>(P.totalKmers /
                                                       (P.singleLutSize
                                                            ? P.singleLutSize
                                                            : 1),
                                                   1)))
            : 1.0;
    printf("probe implied loads: ~%.1f/query -> %.3f Gload/s (%.1f%% of ceiling)\n",
           bucketLoads + 3, gqps * (bucketLoads + 3),
           100.0 * gqps * (bucketLoads + 3) / gLoads);
    printf("(>100%% means the upper binary-search levels and LUT/signature\n"
           " tables are cache-resident: fewer DRAM loads per query than the\n"
           " naive count assumes)\n");

    cudaStreamDestroy(stream);
    cudaFree(d_queries);
    cudaFree(d_hits);
    cudaFree(d_gups);
    cudaFree(d_sink);
    return 0;
}
