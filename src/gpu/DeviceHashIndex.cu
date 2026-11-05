#include "DeviceHashIndex.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cub/cub.cuh>

namespace
{

void cudaCheck(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
}

// atomicCAS twin of fastibs::hashInsertHost. Grid-stride: a thread may
// insert many keys, so a successful insert breaks the probe loop and
// continues with the thread's next key (an early `return` here would
// silently drop every key after the thread's first). The walk is bounded
// by the capacity: a full table — which the sizing margin makes practically
// impossible — raises `overflow` instead of spinning forever.
__global__ void insertKernel(const uint64_t *__restrict__ keys, size_t n,
                             uint64_t *__restrict__ table,
                             fastibs::HashParams P,
                             uint32_t *__restrict__ overflow)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
    {
        const uint64_t code = keys[i];
        uint64_t slot = fastibs::hashSlot0(code, P);
        const uint64_t step = fastibs::hashStep(code, P);
        for (uint64_t tries = 0;; ++tries)
        {
            const uint64_t prev = atomicCAS(
                (unsigned long long *)&table[slot],
                (unsigned long long)fastibs::HASH_EMPTY,
                (unsigned long long)code);
            if (prev == fastibs::HASH_EMPTY || prev == code)
                break;
            if (tries + 1 == P.capacity)
            {
                atomicOr(overflow, 1u);
                break;
            }
            slot += step;
            if (slot >= P.capacity)
                slot -= P.capacity;
        }
    }
}

// One query per thread, grid-stride; the membership test is the shared
// host/device fastibs::hashContains (a query of another shard is 0).
__global__ void hashProbeKernel(const uint64_t *__restrict__ queries, size_t n,
                                const uint64_t *__restrict__ table,
                                fastibs::HashParams P,
                                uint8_t *__restrict__ hits)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        hits[i] = fastibs::hashContains(__ldg(queries + i), table, P) ? 1 : 0;
}

} // namespace

DeviceHashIndex::DeviceHashIndex(const KmcRaIndex &host, double loadFactor,
                                 int device)
    : DeviceHashIndex(host, loadFactor, device, 0, 1)
{
}

DeviceHashIndex::DeviceHashIndex(const KmcRaIndex &host, double loadFactor,
                                 int device, uint32_t shard, uint32_t nShards)
    : dev(device)
{
    if (!(loadFactor > 0.0 && loadFactor < 1.0))
        throw std::invalid_argument(
            "DeviceHashIndex: load factor must be in (0, 1)");
    if (nShards == 0 || shard >= nShards)
        throw std::invalid_argument("DeviceHashIndex: bad shard index");
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");

    k = host.getParams().kmerLength;
    const uint64_t total = host.getParams().totalKmers;
    // Sizing: a shard's key count is known only after the enumeration, but
    // hashShardOf spreads the keys binomially around total/nShards, so the
    // mean plus an 8-sigma margin bounds it in practice (the insert kernel
    // fails loudly, not silently, in the impossible case). The whole set
    // is sized exactly, as before.
    const double mean = (double)total / nShards;
    const uint64_t expected =
        nShards == 1 ? total : (uint64_t)(mean + 8.0 * std::sqrt(mean)) + 1024;
    const uint64_t capacity = fastibs::hashCapacityFor(expected, loadFactor);
    params.capacity = capacity;
    params.shard = shard;
    params.nShards = nShards;
    footprint = capacity * 8;

    const auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t *d_keys = nullptr;
    uint32_t *d_overflow = nullptr;
    uint64_t inserted = 0;
    uint32_t overflow = 0;
    try
    {
        cudaCheck(cudaMalloc(&d_table, capacity * 8), "hash table alloc");
        cudaCheck(cudaMemset(d_table, 0xFF, capacity * 8), "hash table clear");
        cudaCheck(cudaMalloc(&d_overflow, 4), "overflow flag alloc");
        cudaCheck(cudaMemset(d_overflow, 0, 4), "overflow flag clear");

        // Chunked build: enumerate the index's records on the host, upload
        // and insert 32M keys at a time — only this shard's keys; the other
        // shards' devices hold the rest.
        const size_t CHUNK = 32ull << 20;
        cudaCheck(cudaMalloc(&d_keys, CHUNK * 8), "key chunk alloc");
        std::vector<uint64_t> chunk;
        chunk.reserve(CHUNK);
        auto flush = [&]()
        {
            if (chunk.empty())
                return;
            cudaCheck(cudaMemcpy(d_keys, chunk.data(), chunk.size() * 8,
                                 cudaMemcpyHostToDevice),
                      "key chunk H2D");
            const int block = 256;
            const int grid =
                (int)std::min<size_t>((chunk.size() + block - 1) / block, 65535);
            insertKernel<<<grid, block>>>(d_keys, chunk.size(), d_table, params,
                                          d_overflow);
            cudaCheck(cudaGetLastError(), "insert kernel launch");
            cudaCheck(cudaDeviceSynchronize(), "insert kernel");
            inserted += chunk.size();
            chunk.clear();
        };
        host.forEachKmer([&](uint64_t code)
        {
            if (nShards > 1 && fastibs::hashShardOf(code, nShards) != shard)
                return;
            chunk.push_back(code);
            if (chunk.size() == CHUNK)
                flush();
        });
        flush();
        cudaCheck(cudaMemcpy(&overflow, d_overflow, 4, cudaMemcpyDeviceToHost),
                  "overflow flag D2H");
        cudaFree(d_keys);
        cudaFree(d_overflow);
        d_keys = nullptr;
        d_overflow = nullptr;
    }
    catch (...)
    {
        cudaFree(d_keys);
        cudaFree(d_overflow);
        cudaFree(d_table);
        d_table = nullptr;
        throw;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    if (overflow || (nShards == 1 && inserted != total))
    {
        cudaFree(d_table);
        d_table = nullptr;
        throw std::runtime_error(
            overflow ? "DeviceHashIndex: table overflowed (shard key count above the sizing margin)"
                     : "DeviceHashIndex: enumeration count mismatch");
    }
    keysHeld = inserted;
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (nShards == 1)
        fprintf(stderr,
                "[gpu] hash backend: %llu keys, capacity %llu (prime), load %.3f, "
                "%.2f MB, built in %.1f ms\n",
                (unsigned long long)total, (unsigned long long)capacity,
                (double)total / capacity, footprint / 1048576.0, ms);
    else
        fprintf(stderr,
                "[gpu] device %d: hash shard %u/%u: %llu keys (%.1f%% of the index), "
                "capacity %llu (prime), load %.3f, %.2f MB, built in %.1f ms\n",
                device, shard + 1, nShards, (unsigned long long)inserted,
                100.0 * inserted / std::max<uint64_t>(total, 1),
                (unsigned long long)capacity, (double)inserted / capacity,
                footprint / 1048576.0, ms);
}

std::vector<std::unique_ptr<DeviceHashIndex>>
DeviceHashIndex::buildShards(const KmcRaIndex &host, double loadFactor,
                             int nDevices)
{
    nDevices = std::max(1, nDevices);
    std::vector<std::unique_ptr<DeviceHashIndex>> shards(nDevices);
    if (nDevices == 1)
    {
        shards[0] = std::make_unique<DeviceHashIndex>(host, loadFactor, 0, 0, 1);
        return shards;
    }
    // One thread per device: each enumerates the (read-only) host index,
    // keeps its own keys and builds on its own device, all concurrently.
    std::vector<std::exception_ptr> errors(nDevices);
    std::vector<std::thread> workers;
    for (int d = 0; d < nDevices; ++d)
        workers.emplace_back([&, d]
        {
            try
            {
                shards[d] = std::make_unique<DeviceHashIndex>(
                    host, loadFactor, d, (uint32_t)d, (uint32_t)nDevices);
            }
            catch (...)
            {
                errors[d] = std::current_exception();
            }
        });
    for (auto &w : workers)
        w.join();
    for (auto &e : errors)
        if (e)
            std::rethrow_exception(e);

    // Every key of the index on exactly one device (the owner function is
    // a partition, so the shard counts must add up to the total).
    uint64_t held = 0;
    for (const auto &s : shards)
        held += s->keys();
    if (held != host.getParams().totalKmers)
        throw std::runtime_error(
            "DeviceHashIndex: shards hold " + std::to_string(held) +
            " keys, the index has " +
            std::to_string(host.getParams().totalKmers));
    return shards;
}

DeviceHashIndex::~DeviceHashIndex()
{
    cudaSetDevice(dev); // best effort; the free must target this shard's device
    cudaFree(d_table);
    cudaFree(d_stageQueries);
    cudaFree(d_stageHits);
}

void DeviceHashIndex::ensureStage(size_t n)
{
    if (n <= stageCapacity)
        return;
    cudaFree(d_stageQueries);
    cudaFree(d_stageHits);
    d_stageQueries = nullptr;
    d_stageHits = nullptr;
    stageCapacity = std::max(n, stageCapacity * 2);
    cudaCheck(cudaMalloc(&d_stageQueries, stageCapacity * sizeof(uint64_t)),
              "hash stage query alloc");
    cudaCheck(cudaMalloc(&d_stageHits, stageCapacity), "hash stage hit alloc");
}

uint64_t DeviceHashIndex::probeCountHost(const std::vector<uint64_t> &queries)
{
    if (queries.empty())
        return 0;
    std::lock_guard<std::mutex> lock(stageMutex);
    cudaCheck(cudaSetDevice(dev), "cudaSetDevice");
    ensureStage(queries.size());
    cudaCheck(cudaMemcpy(d_stageQueries, queries.data(),
                         queries.size() * sizeof(uint64_t),
                         cudaMemcpyHostToDevice),
              "hash query H2D");
    const int block = 256;
    const int grid = (int)std::min<size_t>((queries.size() + block - 1) / block, 65535);
    hashProbeKernel<<<grid, block>>>(d_stageQueries, queries.size(), d_table, params,
                                     d_stageHits);
    cudaCheck(cudaGetLastError(), "hash probe launch");
    // Device-side reduction: only the 8-byte count crosses PCIe back.
    uint64_t *d_sum = nullptr;
    void *d_temp = nullptr;
    uint64_t sum = 0;
    try
    {
        size_t tempBytes = 0;
        cudaCheck(cudaMalloc(&d_sum, sizeof(uint64_t)), "sum alloc");
        cudaCheck(cub::DeviceReduce::Sum(nullptr, tempBytes, d_stageHits, d_sum,
                                         queries.size()),
                  "reduce temp query");
        cudaCheck(cudaMalloc(&d_temp, tempBytes), "reduce temp alloc");
        cudaCheck(cub::DeviceReduce::Sum(d_temp, tempBytes, d_stageHits, d_sum,
                                         queries.size()),
                  "reduce");
        cudaCheck(cudaMemcpy(&sum, d_sum, sizeof(uint64_t),
                             cudaMemcpyDeviceToHost),
                  "sum D2H");
    }
    catch (...)
    {
        cudaFree(d_temp);
        cudaFree(d_sum);
        throw;
    }
    cudaFree(d_temp);
    cudaFree(d_sum);
    return sum;
}
