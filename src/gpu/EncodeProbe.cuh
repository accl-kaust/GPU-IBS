#pragma once

// Phase 2/3 streaming engine: a ring of `depth` slots, each owning pinned
// host staging buffers, device batch buffers, CUDA streams and events.
// A batch of windows moves through pack (host thread pool) -> H2D ->
// probe kernels -> D2H (bit-packed hit + validity masks), with the copies
// and kernels of consecutive batches overlapping on the different streams.
//
// Phase 5, several devices: the index is split across them (one shard per
// device, DeviceKmerIndex / DeviceHashIndex) and ONE engine drives them
// all. Each batch goes to ONE device — its ENCODER, chosen round-robin so
// every device encodes 1/N of the batches — which runs K1 over the whole
// batch, works out each valid position's owner from its canonical code
// (the prefix range for the ported index, the owner hash for the hash
// backend) and partitions the (code, position) pairs into one contiguous
// segment per shard. The other devices' probe kernels then read their
// segment straight out of the encoder's memory and write one hit flag per
// query back into it — over NVLink where the devices have it, PCIe
// peer-to-peer otherwise; when a pair has no peer access the engine stages
// the copies through host memory instead. Stream/event dependencies chain
// encoder -> owners -> encoder; the encoder finally scatters the flags into
// the position-indexed hit mask and returns hit + validity masks to the
// host over the link the batch came in on. The host side is therefore the
// single-GPU pipeline: one hit mask per batch, nothing to merge, and no
// routing decision taken on the host. Per device the work is encode(1/N of
// the batches) + probe(1/N of the queries).
//
// Probe configurations (the paper's Phase 3 ablation axes):
//  - backend: the ported KMC index (GPU-BS) or the hash set (GPU-HASH)
//  - ordering: fused single kernel (queries live in registers), or the
//    sorted path (encode -> CUB radix sort by k-mer value with position
//    payload -> probe in sorted order, scattering hits back) which trades
//    extra device traffic for memory-coalesced searches (GPU-*-SORTED).
//    The sorted path is a single-device option.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "DeviceHashIndex.cuh"
#include "DeviceKmerIndex.cuh"
#include "EncodeProbeCore.hpp"

class EncodeProbeEngine
{
public:
    // Pinned host-side staging of one slot: pack into seqWords/invWords/
    // descs; read results from hitWords/validWords after wait().
    struct HostSlot
    {
        uint64_t *seqWords;
        uint64_t *invWords;
        fastibs::WindowDesc *descs;
        const uint32_t *hitWords; // database 0's mask (the only one with one database)
        const uint32_t *validWords;
        // Panel mode: database d's hit mask (database-major, one stride of
        // output words per database; the validity mask is shared).
        size_t hitStride;
        uint32_t nDbs;
        const uint32_t *hits(uint32_t d) const { return hitWords + (size_t)d * hitStride; }
    };

    // Capacities of every slot, in units of the batch layout.
    using Caps = fastibs::BatchCaps;

    // The probe structure as one shard per device: exactly one of `bs` /
    // `hash` is non-empty, and shard i is resident on its own device
    // (device()). One entry holding the whole index is the single-GPU
    // case. sortQueries selects the sorted three-stage path (one device).
    // pinLut applies an L2 persisting access policy to each BS shard's LUT
    // on that device's slot streams (no-op for hash).
    EncodeProbeEngine(const std::vector<const DeviceKmerIndex *> &bs,
                      const std::vector<const DeviceHashIndex *> &hash,
                      uint32_t kmerSize, size_t maxBatchBases,
                      size_t maxBatchWindows, int depth, bool sortQueries,
                      bool pinLut = false);
    // Panel mode (docs/MULTIDB_SCAN.md): bs[i][d] is shard d of database i,
    // resident on device d; every database is split with the same cut
    // (KmcRaIndex::jointShardPlan), shares k, and there are at most
    // fastibs::MAX_DBS of them. Each batch is encoded once and probed
    // against every database; the slot returns one hit mask per database
    // and one validity mask. Ported index only, fused ordering only.
    EncodeProbeEngine(const std::vector<std::vector<const DeviceKmerIndex *>> &bs,
                      uint32_t kmerSize, size_t maxBatchBases,
                      size_t maxBatchWindows, int depth, bool pinLut = false);
    ~EncodeProbeEngine();
    EncodeProbeEngine(const EncodeProbeEngine &) = delete;
    EncodeProbeEngine &operator=(const EncodeProbeEngine &) = delete;

    int depth() const { return nSlots; }
    int shards() const { return nDevs; }
    int databases() const { return nDbs; }
    const Caps &caps() const { return cap; }
    HostSlot host(int slot) const;
    // Several devices: owners read the encoder's memory directly (peer
    // access on every pair) rather than through host-staged copies.
    bool peerDirect() const { return direct; }
    // Device memory the engine's buffers take on each device (all slots).
    size_t deviceBufferBytes() const { return perDeviceBytes; }
    // The same figure for parameters an engine has not been built with yet
    // (the constructor's allocations, computed without allocating), so the
    // caller can shrink the batch until the buffers fit next to the shard
    // already resident on the device. peerDirect is what peerAccessible()
    // says about the devices; with several devices the routed path needs
    // 20 + N bytes per batch position per slot (21 with peer access). nDbs
    // databases (panel mode) add a hit mask and a flag lane per database.
    static size_t estimateDeviceBytes(int nDevs, bool peerDirect, size_t maxBatchBases,
                                      size_t maxBatchWindows, int depth, bool sortQueries,
                                      int nDbs = 1);
    // Whether every pair of these devices can access each other's memory
    // (what the constructor's enablePeerAccess() will find); enables nothing.
    // False whenever the staged routing is forced.
    static bool peerAccessible(const std::vector<int> &devices);

    // Forces the host-staged routing (no peer access on any pair) even
    // where the hardware has peer access, so that path can be checked on a
    // peer-capable node: --no-peer, or FASTIBS_NO_PEER=1 in the environment.
    static void forceStaged();
    static bool stagedForced();

    // Async: H2D of the slot's staging to this batch's encoder device,
    // the kernels (routed across the devices when there are several), D2H
    // of the hit and validity masks into the slot's host buffers, then the
    // slot's completion event. The slot's staging buffers must not be
    // repacked until wait(slot) returns.
    void submit(int slot, uint32_t nWindows, size_t seqWordsUsed,
                size_t invWordsUsed, size_t outWordsUsed);

    // Blocks until the slot's in-flight batch (if any) has completed on
    // every device involved (results resident in the slot's host buffers).
    void wait(int slot);

private:
    struct Slot;
    struct Dev;
    struct DevSlot;
    Slot *slots = nullptr;
    int nSlots = 0;
    Dev *devs = nullptr;
    int nDevs = 0;
    int nDbs = 1;
    Caps cap{};
    size_t maxPositions = 0;
    uint32_t k = 0;
    bool sorted = false;
    bool useHash = false;
    bool direct = false;
    size_t perDeviceBytes = 0;
    int nextEncoder = 0;
    // owner function of the ported index: shard d holds prefixes
    // [cutHi[d-1], cutHi[d]); prefix = code >> ownerShift
    uint32_t ownerShift = 0;
    uint32_t ownerCutHi[16] = {};

    void init(const std::vector<std::vector<const DeviceKmerIndex *>> &bs,
              const std::vector<const DeviceHashIndex *> &hash, bool pinLut);
    bool enablePeerAccess();
    void submitSingle(int slot, uint32_t nWindows, size_t seqWordsUsed,
                      size_t invWordsUsed, size_t outWordsUsed);
    void submitRouted(int slot, uint32_t nWindows, size_t seqWordsUsed,
                      size_t invWordsUsed, size_t outWordsUsed);
    void release();
};
