#pragma once

// Device-resident hash-set probe backend (Phase 3, GPU-HASH): an open-
// addressing table of full canonical k-mer codes, built once per database
// from the ported index's records (no KMC rebuild) via chunked uploads and
// an atomicCAS insert kernel. Exposes the same conceptual probe interface
// as DeviceKmerIndex; the fused encode+probe kernel selects the backend.
//
// Phase 5: one object is one shard of the set on one device — the keys
// fastibs::hashShardOf assigns to it, in a table sized for its share — so
// N devices hold ~1/N of the set each and a key is probed for real on
// exactly one of them. One device holds the whole set.

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "HashCore.hpp"
#include "KmcRaIndex.hpp"

class DeviceHashIndex
{
public:
    struct DeviceView
    {
        const uint64_t *table;
        fastibs::HashParams params;
        uint32_t k;
    };

    // Builds the whole set on `device`; logs build time, capacity, load
    // factor and footprint. Throws on CUDA failure.
    DeviceHashIndex(const KmcRaIndex &host, double loadFactor, int device = 0);
    // Builds shard `shard` of `nShards` on `device`.
    DeviceHashIndex(const KmcRaIndex &host, double loadFactor, int device,
                    uint32_t shard, uint32_t nShards);
    ~DeviceHashIndex();
    DeviceHashIndex(const DeviceHashIndex &) = delete;
    DeviceHashIndex &operator=(const DeviceHashIndex &) = delete;

    // Builds shard d on device d for d < nDevices, concurrently (one host
    // thread per device), and checks that the shards hold every key of the
    // index exactly once. nDevices == 1: the whole set on device 0.
    static std::vector<std::unique_ptr<DeviceHashIndex>>
    buildShards(const KmcRaIndex &host, double loadFactor, int nDevices);

    // Counts the queries present in this shard (queries of other shards
    // answer 0 by construction): the batched probe primitive of the
    // intersection tool, same contract as DeviceKmerIndex::probeCountHost.
    uint64_t probeCountHost(const std::vector<uint64_t> &queries);

    DeviceView view() const { return {d_table, params, k}; }
    int device() const { return dev; }
    uint64_t keys() const { return keysHeld; }
    size_t footprintBytes() const { return footprint; }

private:
    int dev = 0;
    uint64_t *d_table = nullptr;
    fastibs::HashParams params{};
    uint32_t k = 0;
    uint64_t keysHeld = 0;
    size_t footprint = 0;

    std::mutex stageMutex;
    uint64_t *d_stageQueries = nullptr;
    uint8_t *d_stageHits = nullptr;
    size_t stageCapacity = 0;
    void ensureStage(size_t n);
};
