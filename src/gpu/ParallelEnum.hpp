#pragma once

// Multi-threaded enumeration of a KMC database's k-mers feeding a batched
// membership probe — the host side of KDBIntersect.
//
// The enumeration (KmcRaIndex::forEachKmer) walks the index's look-up
// tables bucket by bucket and reconstructs each k-mer from the bucket's
// prefix and its suffix record. Single-threaded, it was the whole wall time
// of the GPU intersection: the 2e9 device probes of the self-intersection
// take ~1.5 s of a 17.6 s run, so
// the tool was host-bound and no probe structure could show its worth.
//
// Here the walk is cut into UNITS — contiguous ranges of whole buckets
// holding about one batch of records each (planEnumUnits) — that N
// producer threads take from a shared counter. A producer reconstructs its
// unit's k-mers into its own batch buffers, one per SINK (a sink is a
// device shard: the host routes a k-mer to the shard holding it, as the
// sequential code did), and hands each full batch to the sink's bounded
// queue. Consumer threads pop the batches and call the probe — on the GPU
// path DeviceKmerIndex/DeviceHashIndex::probeCountHost, which serializes
// on its own staging buffers, so one consumer per device; on the CPU path
// the read-only host probe, so one consumer per thread — and the
// per-batch counts are summed.
//
// The count is exactly the sequential path's: every bucket is in exactly
// one unit, a unit is walked exactly as the sequential walk walks it
// (KmcRaIndex::forEachKmerIn), every k-mer lands in exactly one batch and
// every batch is probed once, and a sum does not depend on the order the
// batches reach a probe. With one producer the units are taken in on-disk
// order and the walk is the sequential one. This header is CUDA-free.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "KmcRaIndex.hpp"

namespace fastibs
{

// Cuts of the enumeration into units: unit u is the buckets
// [cuts[u], cuts[u+1]) of the LUT, about `unitRecords` records each. Along
// the on-disk record order the LUT is non-decreasing (bin after bin, prefix
// after prefix within a bin), so a binary search places each cut. The cuts
// are forced increasing and end at the last bucket, so the units partition
// the buckets whatever the LUT holds; a non-monotone LUT could only
// unbalance them. A bucket larger than a unit is one unit by itself.
inline std::vector<size_t> planEnumUnits(const KmcRaIndex &index, uint64_t unitRecords)
{
    const auto &lut = index.getLut();
    const size_t buckets = index.numBuckets();
    const uint64_t total = index.getParams().totalKmers;
    unitRecords = std::max<uint64_t>(1, unitRecords);
    std::vector<size_t> cuts{0};
    for (uint64_t target = unitRecords; target < total; target += unitRecords)
    {
        const size_t j = (size_t)(std::lower_bound(lut.begin(), lut.begin() + buckets, target) -
                                  lut.begin());
        if (j > cuts.back() && j < buckets)
            cuts.push_back(j);
    }
    cuts.push_back(buckets);
    return cuts;
}

// Bounded hand-off of batches from producers to consumers. Drained
// buffers go back to the producers, so a buffer is page-faulted once, not
// once per batch.
class BatchQueue
{
public:
    using Batch = std::vector<uint64_t>;

    explicit BatchQueue(size_t capacity) : cap(std::max<size_t>(1, capacity)) {}

    // Blocks while the queue is full. False once the queue was aborted.
    bool push(Batch &&b)
    {
        std::unique_lock<std::mutex> lk(m);
        notFull.wait(lk, [&] { return aborted || q.size() < cap; });
        if (aborted)
            return false;
        q.push_back(std::move(b));
        notEmpty.notify_one();
        return true;
    }

    // Blocks while the queue is empty. False once it is closed and drained,
    // or aborted.
    bool pop(Batch &out)
    {
        std::unique_lock<std::mutex> lk(m);
        notEmpty.wait(lk, [&] { return aborted || closed || !q.empty(); });
        if (aborted || q.empty())
            return false;
        out = std::move(q.front());
        q.pop_front();
        notFull.notify_one();
        return true;
    }

    // No more batches will be pushed: consumers finish what is queued.
    void close()
    {
        std::lock_guard<std::mutex> lk(m);
        closed = true;
        notEmpty.notify_all();
    }

    // A thread failed: wake everybody up to stop, queued batches dropped.
    void abort()
    {
        std::lock_guard<std::mutex> lk(m);
        aborted = true;
        notEmpty.notify_all();
        notFull.notify_all();
    }

    void recycle(Batch &&b)
    {
        b.clear(); // keeps the capacity
        std::lock_guard<std::mutex> lk(m);
        spare.push_back(std::move(b));
    }

    Batch take(size_t reserveKeys)
    {
        {
            std::lock_guard<std::mutex> lk(m);
            if (!spare.empty())
            {
                Batch b = std::move(spare.back());
                spare.pop_back();
                return b;
            }
        }
        Batch b;
        b.reserve(reserveKeys);
        return b;
    }

private:
    std::mutex m;
    std::condition_variable notFull, notEmpty;
    std::deque<Batch> q;
    std::vector<Batch> spare;
    const size_t cap;
    bool closed = false;
    bool aborted = false;
};

struct EnumCountStats
{
    unsigned producers = 0;  // enumeration threads
    unsigned consumers = 0;  // probe threads, all sinks together
    size_t units = 0;        // work units the enumeration was cut into
    size_t batches = 0;      // batches handed to the probe
    size_t batchKeys = 0;    // keys per full batch (per sink)
    uint64_t keys = 0;       // k-mers enumerated
    double enumSeconds = 0;  // until the last producer finished
    double wallSeconds = 0;  // until the last batch was probed
    double probeSeconds = 0; // time inside probeBatch, summed over consumers
};

// Enumerates `index` on `producers` threads (the calling thread is one of
// them), routes each k-mer to the sink ownerOf(code) in [0, nSinks) — not
// called when nSinks == 1 — in batches of `batchKeys` keys, and has
// `consumersPerSink` threads per sink call probeBatch(sink, batch), which
// returns how many of the batch's keys are present. Returns the sum over
// every batch. An exception thrown by either callback (or by the index)
// stops every thread and is rethrown here.
template <typename OwnerFn, typename ProbeFn>
uint64_t enumerateAndCount(const KmcRaIndex &index, unsigned producers, int nSinks,
                           size_t batchKeys, unsigned consumersPerSink,
                           OwnerFn &&ownerOf, ProbeFn &&probeBatch,
                           EnumCountStats *stats = nullptr)
{
    using Batch = BatchQueue::Batch;
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::duration d)
    { return std::chrono::duration<double>(d).count(); };

    producers = std::max(1u, producers);
    consumersPerSink = std::max(1u, consumersPerSink);
    nSinks = std::max(1, nSinks);
    batchKeys = std::max<size_t>(1, batchKeys);

    const auto t0 = Clock::now();
    const std::vector<size_t> cuts = planEnumUnits(index, batchKeys);
    const size_t nUnits = cuts.size() - 1;

    // Per sink, the queue holds at most `producers` batches, capped by a
    // 256 MiB byte budget, so the queued buffers stay bounded whatever the
    // thread count; the producers' filling buffers (one per sink each) and
    // the consumers' in-flight batches come on top of that.
    const size_t budgetBatches =
        std::max<size_t>(1, (256ull << 20) / (batchKeys * sizeof(uint64_t)));
    const size_t capacity = std::max<size_t>(1, std::min<size_t>(producers, budgetBatches));
    std::vector<std::unique_ptr<BatchQueue>> queues;
    for (int s = 0; s < nSinks; ++s)
        queues.push_back(std::make_unique<BatchQueue>(capacity));

    std::atomic<size_t> nextUnit{0};
    std::atomic<size_t> nBatches{0};
    std::atomic<uint64_t> nKeys{0};
    std::mutex errMutex;
    std::exception_ptr error;
    auto fail = [&]
    {
        {
            std::lock_guard<std::mutex> lk(errMutex);
            if (!error)
                error = std::current_exception();
        }
        for (auto &q : queues)
            q->abort();
    };
    struct Aborted
    {
    };

    // Consumers: pop, probe, add, give the buffer back.
    const unsigned nConsumers = (unsigned)nSinks * consumersPerSink;
    std::vector<uint64_t> partial(nConsumers, 0);
    std::vector<double> probeSecs(nConsumers, 0);
    auto consume = [&](unsigned c)
    {
        const int sink = (int)(c / consumersPerSink);
        BatchQueue &q = *queues[sink];
        try
        {
            Batch b;
            while (q.pop(b))
            {
                const auto p0 = Clock::now();
                partial[c] += probeBatch(sink, b);
                probeSecs[c] += seconds(Clock::now() - p0);
                q.recycle(std::move(b));
            }
        }
        catch (...)
        {
            fail();
        }
    };

    // Producers: take units until none is left; a full batch goes to its
    // sink's queue, the partial ones at the end too.
    auto produce = [&]
    {
        try
        {
            std::vector<Batch> filling((size_t)nSinks);
            for (int s = 0; s < nSinks; ++s)
                filling[s] = queues[s]->take(batchKeys);
            auto hand = [&](int s)
            {
                const size_t n = filling[s].size();
                if (!queues[s]->push(std::move(filling[s])))
                    throw Aborted{};
                nKeys += n; // counted once the batch is really on its way
                ++nBatches;
                filling[s] = queues[s]->take(batchKeys);
            };
            for (;;)
            {
                const size_t u = nextUnit.fetch_add(1);
                if (u >= nUnits)
                    break;
                index.forEachKmerIn(cuts[u], cuts[u + 1], [&](uint64_t code)
                {
                    const int s = nSinks == 1 ? 0 : (int)ownerOf(code);
                    Batch &f = filling[s];
                    f.push_back(code);
                    if (f.size() == batchKeys)
                        hand(s);
                });
            }
            for (int s = 0; s < nSinks; ++s)
                if (!filling[s].empty())
                    hand(s);
        }
        catch (const Aborted &)
        {
        }
        catch (...)
        {
            fail();
        }
    };
    // Thread creation can fail (std::system_error, e.g. a thread limit):
    // then nothing may be left joinable — abort the queues so the threads
    // that did start return, join them, and rethrow.
    std::vector<std::thread> consumers, workers;
    consumers.reserve(nConsumers);
    workers.reserve(producers - 1);
    auto joinAll = [&]
    {
        for (auto &w : workers)
            if (w.joinable())
                w.join();
        for (auto &t : consumers)
            if (t.joinable())
                t.join();
    };
    try
    {
        for (unsigned c = 0; c < nConsumers; ++c)
            consumers.emplace_back(consume, c);
        for (unsigned p = 1; p < producers; ++p)
            workers.emplace_back(produce);
    }
    catch (...)
    {
        for (auto &q : queues)
            q->abort();
        joinAll();
        throw;
    }
    produce();
    for (auto &w : workers)
        w.join();
    const auto tEnum = Clock::now();

    for (auto &q : queues)
        q->close();
    for (auto &t : consumers)
        t.join();
    const auto tEnd = Clock::now();

    if (error)
        std::rethrow_exception(error);

    uint64_t count = 0;
    for (uint64_t p : partial)
        count += p;
    if (stats)
    {
        double probeTotal = 0;
        for (double s : probeSecs)
            probeTotal += s;
        stats->producers = producers;
        stats->consumers = nConsumers;
        stats->units = nUnits;
        stats->batches = nBatches.load();
        stats->batchKeys = batchKeys;
        stats->keys = nKeys.load();
        stats->enumSeconds = seconds(tEnum - t0);
        stats->wallSeconds = seconds(tEnd - t0);
        stats->probeSeconds = probeTotal;
    }
    return count;
}

} // namespace fastibs
