#include "DeviceKmerIndex.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#include <cub/cub.cuh>

namespace
{

void cudaCheck(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
}

using fastibs::ProbeParams;

// One query per thread, grid-stride. The search itself is the shared
// host/device probeOne from ProbeCore.hpp — the same code validated on the
// host against CKMCFile by tests/phase1_test.cpp.
template <typename RecT>
__global__ void probeKernel(const uint64_t *__restrict__ queries,
                            const uint8_t *__restrict__ valid,
                            size_t n,
                            const uint64_t *__restrict__ lut,
                            const RecT *__restrict__ suffixes,
                            const uint32_t *__restrict__ sigMap,
                            const uint32_t *__restrict__ norm,
                            ProbeParams P,
                            uint8_t *__restrict__ hits)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
    {
        if (valid && !valid[i])
        {
            hits[i] = 0;
            continue;
        }
        const uint64_t code = __ldg(queries + i);
        hits[i] = fastibs::probeOne<RecT>(code, lut, suffixes, sigMap, norm, P) ? 1 : 0;
    }
}

template <typename T, typename A>
T *uploadArray(const std::vector<T, A> &host, const char *what)
{
    if (host.empty())
        return nullptr;
    T *d = nullptr;
    cudaCheck(cudaMalloc(&d, host.size() * sizeof(T)), what);
    cudaCheck(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              what);
    return d;
}

} // namespace

DeviceKmerIndex::DeviceKmerIndex(const KmcRaIndex &host, int device)
    : DeviceKmerIndex(host, device, host.shardPlan(1)[0])
{
}

DeviceKmerIndex::DeviceKmerIndex(const KmcRaIndex &host, int device,
                                 const KmcRaIndex::Shard &shard)
    : dev(device), shardInfo(shard)
{
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");

    cudaDeviceProp prop{};
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    // The shard's image: rebased LUT slice plus, per signature bin, the run
    // of records to gather. For the single shard of a one-device run this
    // is the on-disk layout itself.
    const KmcRaIndex::ShardLayout layout = host.shardLayout(shard);
    params = layout.params;
    lutEntries = layout.lut.size();
    rec32 = host.recordsAre32Bit();
    const size_t recBytes = rec32 ? 4 : 8;

    const auto t0 = std::chrono::high_resolution_clock::now();
    try
    {
        d_lut = uploadArray(layout.lut, "lut upload");
        d_sigMap = uploadArray(host.getSignatureMap(), "signature map upload");
        d_norm = uploadArray(host.getNorm(), "norm table upload");
        if (layout.records)
        {
            cudaCheck(cudaMalloc(&d_suffix, layout.records * recBytes), "suffix alloc");
            // Gather: each bin's run lands right after the previous bin's,
            // one copy per bin straight from the host arrays (one copy in
            // total for KMC1 or a whole-index shard of a single-bin LUT).
            const char *src = rec32 ? (const char *)host.getSuffix32().data()
                                    : (const char *)host.getSuffix64().data();
            size_t off = 0;
            for (const auto &range : layout.ranges)
            {
                const uint64_t lo = range.first, hi = range.second;
                if (hi > lo)
                    cudaCheck(cudaMemcpy((char *)d_suffix + off * recBytes,
                                         src + lo * recBytes, (hi - lo) * recBytes,
                                         cudaMemcpyHostToDevice),
                              "suffix upload");
                off += hi - lo;
            }
        }
    }
    catch (...)
    {
        release();
        throw;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    footprint = lutEntries * 8 + host.getSignatureMap().size() * 4 +
                host.getNorm().size() * 4 + layout.records * recBytes;
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Diagnostics go to stderr: KDBIntersect's stdout carries only the
    // count, and callers parse it.
    if (shard.count == 1)
        fprintf(stderr,
                "[gpu] device %d (%s): index resident, %.2f MB, uploaded in %.1f ms\n",
                device, prop.name, footprint / 1048576.0, ms);
    else
        fprintf(stderr,
                "[gpu] device %d (%s): index shard %d/%d resident: prefixes [%u, %u) of %llu, "
                "%llu records (%.1f%% of the index), %.2f MB, uploaded in %.1f ms\n",
                device, prop.name, shard.index + 1, shard.count, shard.prefixLo,
                shard.prefixHi, (unsigned long long)host.getParams().singleLutSize,
                (unsigned long long)layout.records,
                100.0 * layout.records / std::max<uint64_t>(host.getParams().totalKmers, 1),
                footprint / 1048576.0, ms);
}

std::vector<std::unique_ptr<DeviceKmerIndex>>
DeviceKmerIndex::buildShards(const KmcRaIndex &host, int nDevices)
{
    return buildShards(host, host.shardPlan(std::max(1, nDevices)));
}

std::vector<std::unique_ptr<DeviceKmerIndex>>
DeviceKmerIndex::buildShards(const KmcRaIndex &host,
                             const std::vector<KmcRaIndex::Shard> &plan)
{
    const int nDevices = (int)plan.size();
    if (nDevices < 1)
        throw std::runtime_error("DeviceKmerIndex::buildShards: empty plan");
    std::vector<std::unique_ptr<DeviceKmerIndex>> shards(nDevices);
    if (nDevices == 1)
    {
        shards[0] = std::make_unique<DeviceKmerIndex>(host, 0, plan[0]);
        return shards;
    }
    std::vector<std::exception_ptr> errors(nDevices);
    std::vector<std::thread> workers;
    for (int d = 0; d < nDevices; ++d)
        workers.emplace_back([&, d]
        {
            try
            {
                shards[d] = std::make_unique<DeviceKmerIndex>(host, d, plan[d]);
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
    return shards;
}

DeviceKmerIndex::~DeviceKmerIndex()
{
    cudaSetDevice(dev); // best effort; frees must target this shard's device
    release();
}

void DeviceKmerIndex::release()
{
    cudaFree(d_lut);
    cudaFree(d_suffix);
    cudaFree(d_sigMap);
    cudaFree(d_norm);
    cudaFree(d_stageQueries);
    cudaFree(d_stageHits);
    d_lut = nullptr;
    d_suffix = nullptr;
    d_sigMap = nullptr;
    d_norm = nullptr;
    d_stageQueries = nullptr;
    d_stageHits = nullptr;
    stageCapacity = 0;
}

void DeviceKmerIndex::probe(const uint64_t *d_queries, const uint8_t *d_valid,
                            size_t n, uint8_t *d_hits, cudaStream_t stream) const
{
    if (n == 0)
        return;
    const int block = 256;
    const int grid =
        (int)std::min<size_t>((n + block - 1) / block, 65535);

    if (rec32)
        probeKernel<uint32_t><<<grid, block, 0, stream>>>(
            d_queries, d_valid, n, d_lut, (const uint32_t *)d_suffix, d_sigMap,
            d_norm, params, d_hits);
    else
        probeKernel<uint64_t><<<grid, block, 0, stream>>>(
            d_queries, d_valid, n, d_lut, (const uint64_t *)d_suffix, d_sigMap,
            d_norm, params, d_hits);
    cudaCheck(cudaGetLastError(), "probe kernel launch");
}

// Caller holds stageMutex and has this shard's device set.
void DeviceKmerIndex::ensureStage(size_t n)
{
    if (n <= stageCapacity)
        return;
    cudaFree(d_stageQueries);
    cudaFree(d_stageHits);
    d_stageQueries = nullptr;
    d_stageHits = nullptr;
    stageCapacity = std::max(n, stageCapacity * 2);
    cudaCheck(cudaMalloc(&d_stageQueries, stageCapacity * sizeof(uint64_t)),
              "stage query alloc");
    cudaCheck(cudaMalloc(&d_stageHits, stageCapacity), "stage hit alloc");
}

void DeviceKmerIndex::probeHost(const std::vector<uint64_t> &queries,
                                std::vector<uint8_t> &hits)
{
    hits.assign(queries.size(), 0);
    if (queries.empty())
        return;

    std::lock_guard<std::mutex> lock(stageMutex);
    cudaCheck(cudaSetDevice(dev), "cudaSetDevice");
    ensureStage(queries.size());

    cudaCheck(cudaMemcpy(d_stageQueries, queries.data(),
                         queries.size() * sizeof(uint64_t),
                         cudaMemcpyHostToDevice),
              "query H2D");
    probe(d_stageQueries, nullptr, queries.size(), d_stageHits, 0);
    cudaCheck(cudaMemcpy(hits.data(), d_stageHits, hits.size(),
                         cudaMemcpyDeviceToHost),
              "hits D2H");
}

uint64_t DeviceKmerIndex::probeCountHost(const std::vector<uint64_t> &queries)
{
    if (queries.empty())
        return 0;
    std::lock_guard<std::mutex> lock(stageMutex);
    cudaCheck(cudaSetDevice(dev), "cudaSetDevice");
    ensureStage(queries.size());

    cudaCheck(cudaMemcpy(d_stageQueries, queries.data(),
                         queries.size() * sizeof(uint64_t),
                         cudaMemcpyHostToDevice),
              "query H2D");
    probe(d_stageQueries, nullptr, queries.size(), d_stageHits, 0);

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

namespace
{
std::string g_deviceCountError = "not queried";
}

int DeviceKmerIndex::deviceCount()
{
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess)
    {
        // Keep the reason for the callers message. Typical strings:
        // "no CUDA-capable device is detected" (no GPU, or no driver library
        // -- e.g. a container started without --nv), "CUDA driver version is
        // insufficient for CUDA runtime version" (old driver), "unknown
        // error" (a broken device; SLURM own NVML calls fail on such a node).
        g_deviceCountError = cudaGetErrorString(err);
        if (err == cudaErrorInsufficientDriver)
        {
            // The runtime says "insufficient driver" both for an old driver
            // and for none at all; cudaDriverGetVersion tells them apart
            // (0 = no driver library could be loaded), and the second is by
            // far the common case: a CPU-only machine, or a container
            // started without --gpus all (Docker) / --nv (Apptainer).
            int driver = 0;
            cudaDriverGetVersion(&driver);
            if (driver == 0)
                g_deviceCountError += "; no NVIDIA driver is loaded: no GPU on this machine, "
                                      "or a container started without --gpus all / --nv";
            else
                g_deviceCountError += "; the driver supports CUDA " + std::to_string(driver / 1000) + "." +
                                      std::to_string(driver % 1000 / 10) +
                                      ", these tools need CUDA 12 (driver >= 525.60)";
        }
        cudaGetLastError(); // clear the sticky error state
        return 0;
    }
    g_deviceCountError = n == 0 ? "the driver reports 0 devices" : "";
    return n;
}

const char *DeviceKmerIndex::deviceCountError()
{
    return g_deviceCountError.c_str();
}

size_t DeviceKmerIndex::freeMemory(int device)
{
    int prev = 0;
    cudaGetDevice(&prev);
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
    {
        cudaGetLastError();
        freeBytes = 0;
    }
    cudaSetDevice(prev);
    return freeBytes;
}

void DeviceKmerIndex::setLutPersisting(cudaStream_t stream) const
{
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (prop.persistingL2CacheMaxSize <= 0)
    {
        fprintf(stderr,
                "[gpu] LUT pinning requested but the device has no persisting L2\n");
        return;
    }
    // The window must span the FULL LUT slice: for KMC2 the queries index
    // num_bins * stride entries spread across all bins by signature, so
    // pinning only one bin's slice would measure nothing. When the LUT
    // exceeds the persisting carve, a fractional hitRatio pins a uniform
    // sample of it.
    const size_t lutBytes = lutEntries * 8;
    const size_t carve = std::min<size_t>((size_t)prop.persistingL2CacheMaxSize,
                                          lutBytes);
    cudaCheck(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve),
              "persisting L2 limit");

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = (void *)d_lut;
    attr.accessPolicyWindow.num_bytes =
        std::min<size_t>(lutBytes, (size_t)prop.accessPolicyMaxWindowSize);
    attr.accessPolicyWindow.hitRatio =
        (float)std::min(1.0, (double)carve / (double)attr.accessPolicyWindow.num_bytes);
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaCheck(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                                     &attr),
              "LUT access policy");
    fprintf(stderr,
            "[gpu] LUT persisting-L2 window: %.1f MB of %.1f MB LUT, hit ratio %.2f\n",
            attr.accessPolicyWindow.num_bytes / 1048576.0, lutBytes / 1048576.0,
            attr.accessPolicyWindow.hitRatio);
}
