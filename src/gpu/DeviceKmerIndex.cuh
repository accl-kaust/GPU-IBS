#pragma once

// Device-resident KMC random-access index and batched membership probe
// (Phase 1). The index is a counter-stripped, integer-normalized copy of
// the on-disk structure loaded by KmcRaIndex; the probe kernel mirrors
// KmcRaIndex::probe (and therefore CKMCFile::CheckKmer) operation for
// operation, so device results are bit-exact with the CPU path.
//
// Phase 5: one object is one SHARD of the index on one device. With a
// single device the shard is the whole index, uploaded verbatim; with N
// devices, device d holds prefix range d of KmcRaIndex::shardPlan (~1/N of
// the records and of the LUT) and probes queries outside its range as
// misses, so every query is searched on exactly one device and the
// per-device footprint is ~1/N of the index.

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "KmcRaIndex.hpp"
#include "ProbeCore.hpp"

class DeviceKmerIndex
{
public:
    // Raw device pointers + query-path constants, for kernels defined in
    // other translation units (the Phase 2 fused encode+probe kernel).
    struct DeviceView
    {
        const uint64_t *lut;
        const void *suffix; // const uint32_t* when rec32, else const uint64_t*
        const uint32_t *sigMap;
        const uint32_t *norm;
        bool rec32;
        fastibs::ProbeParams params;
    };
    DeviceView view() const { return {d_lut, d_suffix, d_sigMap, d_norm, rec32, params}; }

    // Uploads the whole index to `device` (the single-GPU configuration).
    explicit DeviceKmerIndex(const KmcRaIndex &host, int device = 0);
    // Uploads one shard of the index (an entry of host.shardPlan(n)) to
    // `device`; logs upload time and footprint. Throws std::runtime_error
    // on CUDA failure.
    DeviceKmerIndex(const KmcRaIndex &host, int device, const KmcRaIndex::Shard &shard);
    ~DeviceKmerIndex();
    DeviceKmerIndex(const DeviceKmerIndex &) = delete;
    DeviceKmerIndex &operator=(const DeviceKmerIndex &) = delete;

    // Splits the index into nDevices shards and uploads shard d to device
    // d, concurrently — one host thread per device (the CUDA device is
    // per-thread state, and the uploads then run over the devices' links
    // in parallel). nDevices == 1: the whole index on device 0.
    static std::vector<std::unique_ptr<DeviceKmerIndex>>
    buildShards(const KmcRaIndex &host, int nDevices);
    // The same with a given plan (shard d to device d): panel mode uploads
    // every database of a round with one joint cut
    // (KmcRaIndex::jointShardPlan), so the device-side owner function is
    // the same for all of them.
    static std::vector<std::unique_ptr<DeviceKmerIndex>>
    buildShards(const KmcRaIndex &host, const std::vector<KmcRaIndex::Shard> &plan);

    int device() const { return dev; }
    const KmcRaIndex::Shard &shard() const { return shardInfo; }

    // Async batched probe: d_queries[n] are canonical 2-bit packed k-mers;
    // writes d_hits[n] in {0,1}. d_valid may be null; when given, positions
    // with d_valid[i] == 0 get d_hits[i] = 0 without probing. Queries held
    // by another shard probe as 0. The caller has this shard's device set.
    void probe(const uint64_t *d_queries, const uint8_t *d_valid, size_t n,
               uint8_t *d_hits, cudaStream_t stream) const;

    // Synchronous convenience used by the Phase 1 per-window batched path:
    // stages queries through internally managed device buffers (grown on
    // demand), probes, and fills hits. Thread-safe (calls are serialized);
    // sets the calling thread's device to this shard's.
    void probeHost(const std::vector<uint64_t> &queries, std::vector<uint8_t> &hits);

    // Phase 4 intersection primitive: probes the queries and returns how
    // many are present in this shard (device-side reduction; only the count
    // comes back). Same staging buffers as probeHost; thread-safe.
    uint64_t probeCountHost(const std::vector<uint64_t> &queries);

    // Number of usable CUDA devices; 0 when no device or no driver.
    static int deviceCount();
    // Why the last deviceCount() returned 0: the cudaGetDeviceCount error
    // string (empty when devices were found).
    static const char *deviceCountError();
    // Free memory of `device` in bytes; 0 when it cannot be queried.
    static size_t freeMemory(int device);

    // Phase 3 ablation: mark the prefix LUT as L2-persisting for kernels
    // launched on `stream` (clamped to the device's persisting-L2 and
    // access-policy-window limits). Measures whether the two LUT loads per
    // query matter once other traffic is coalesced. Caller has this
    // shard's device set.
    void setLutPersisting(cudaStream_t stream) const;

    size_t footprintBytes() const { return footprint; }

private:
    int dev = 0;
    KmcRaIndex::Shard shardInfo{};
    fastibs::ProbeParams params{}; // this shard's query-path constants
    size_t lutEntries = 0; // this shard's LUT slice (KmcRaIndex::shardLayout)
    bool rec32 = false;

    uint64_t *d_lut = nullptr;
    void *d_suffix = nullptr; // uint32_t* when rec32, else uint64_t*
    uint32_t *d_sigMap = nullptr;
    uint32_t *d_norm = nullptr;
    size_t footprint = 0;

    // staging buffers for probeHost / probeCountHost
    std::mutex stageMutex;
    uint64_t *d_stageQueries = nullptr;
    uint8_t *d_stageHits = nullptr;
    size_t stageCapacity = 0;

    void ensureStage(size_t n);
    void release();
};
