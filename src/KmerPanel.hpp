#pragma once

// Panel mode: N sample databases scanned against a reference in ONE
// reference pass (docs/MULTIDB_SCAN.md). The reference is read and windowed
// once; the databases are taken in rounds of as many as fit the device(s)
// (or --resident=N); within a round every packed batch is encoded once on
// the GPU and probed against every resident database, and on the CPU every
// window's canonical codes are computed once and probed against each
// database; the per-window statistics and the TSV files come out per
// database, byte-identical to N single runs (the same Window records
// through the same writer).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "KmerDatabase.hpp"

class KmerPanel
{
public:
    // prefixes[i]: database i's KMC path prefix (folder + "/" + prefix, as
    // fastibs resolves it); names[i]: its output name (the folder's name).
    // resident: at most this many databases per round (0: as many as fit).
    KmerPanel(const vector<string> &prefixes, const vector<string> &names, int resident)
        : residentCap(resident)
    {
        if (prefixes.empty() || prefixes.size() != names.size())
            throw std::runtime_error("KmerPanel: no databases");
        for (size_t i = 0; i < prefixes.size(); i++)
        {
            Member m;
            m.prefix = prefixes[i];
            m.name = names[i];
            m.db = std::make_unique<KmerDatabase>(prefixes[i]);
            if (!members.empty() && m.db->getKmerSize() != members[0].db->getKmerSize())
                throw std::runtime_error("KmerPanel: " + m.name + " has k=" +
                                         std::to_string(m.db->getKmerSize()) + " but " +
                                         members[0].name + " has k=" +
                                         std::to_string(members[0].db->getKmerSize()) +
                                         "; a panel shares one k");
            // Every panel path packs a k-mer into 64 bits; fastibs runs a
            // k > 32 panel as single scans on the string path instead.
            if (KmerDatabase::needsStringPath(m.db->getKmerSize()))
                throw std::runtime_error("KmerPanel: k=" + std::to_string(m.db->getKmerSize()) +
                                         " > 32 has no panel path");
            members.push_back(std::move(m));
        }
    }

    size_t size() const { return members.size(); }
    const string &name(size_t i) const { return members[i].name; }
    uint kmerSize() const { return members[0].db->getKmerSize(); }
    KmerDatabase &database(size_t i) { return *members[i].db; }

    // One reference against every member whose outPaths[i] is non-empty
    // (an empty path skips that member, e.g. its file already exists).
    void processReference(const string &refPath, const vector<string> &outPaths, int windowSize)
    {
        if (outPaths.size() != members.size())
            throw std::runtime_error("KmerPanel: one output path per database expected");
        vector<size_t> active;
        for (size_t i = 0; i < members.size(); i++)
            if (!outPaths[i].empty())
                active.push_back(i);
        if (active.empty())
            return;

        const auto tRef = chrono::steady_clock::now();
        auto secondsSince = [](const chrono::steady_clock::time_point &t)
        { return chrono::duration<double>(chrono::steady_clock::now() - t).count(); };

        const uint k = kmerSize();
        cout << "Window size: " << windowSize << endl;
        cout << "Kmer size: " << k << endl;
        if (KmerDatabase::legacyRequested())
            cout << "[panel] FASTIBS_LEGACY has no panel path; using the rolling codec" << endl;

        bool gpu = false;
        int nDev = 0;
#ifdef FASTIBS_CUDA
        if (KmerDatabase::gpuRequested())
        {
            const int avail = DeviceKmerIndex::deviceCount();
            if (avail == 0)
            {
                cout << "[gpu] no CUDA device found, using CPU path (cudaGetDeviceCount: "
                     << DeviceKmerIndex::deviceCountError() << ")" << endl;
                KmerDatabase::warnGpuForcedWithoutDevice();
            }
            else
            {
                gpu = true;
                nDev = KmerDatabase::devicesToUse(avail);
                const auto &o = KmerDatabase::gpuOptions();
                if (o.probeHash || o.sortQueries)
                    throw std::runtime_error(
                        "panel mode probes through the ported index in fused order: "
                        "--probe=hash and --sort-queries are single-database options");
                if (KmerDatabase::gpuSyncRequested())
                    cout << "[panel] --sync has no panel path; using the streamed pipeline" << endl;
            }
        }
#endif
        cout << "K-mer path: "
             << (gpu ? "fused GPU encode+probe, streamed (GPU-BS), " +
                           std::to_string(active.size()) + " databases per encode"
                     : "rolling codec (CPU-OPT), " + std::to_string(active.size()) +
                           " databases per window")
             << endl;

        // The reference: read and windowed once for every round.
        KmerDatabase::ReferenceWindows ref;
        members[active[0]].db->readReferenceWindows(refPath, windowSize, ref);
        const auto &chunks = ref.chunks;

        // Rounds: resident sets in command-line order. When the previous
        // reference ran as one round over the same databases, its images
        // (or random-access views) are still resident and are reused, as a
        // single-database run reuses its image across references; anything
        // else is released first, so planning sees the device's free memory
        // and a new upload never lands on top of an old image.
        vector<vector<size_t>> rounds;
        if (!keptResident.empty() && keptResident == active)
            rounds = {active};
        else
        {
            releaseAll();
            rounds = planRounds(active, gpu, nDev, chunks);
        }
        {
            cout << "[panel] " << active.size() << " databases in " << rounds.size()
                 << (rounds.size() == 1 ? " round" : " rounds");
            if (rounds.size() > 1)
            {
                cout << " (";
                for (size_t r = 0; r < rounds.size(); r++)
                    cout << (r ? ", " : "") << rounds[r].size();
                cout << " databases)";
            }
            cout << endl;
        }

        double setupTotal = 0, statsTotal = 0, writeTotal = 0;
        string writeErrors;
        for (size_t r = 0; r < rounds.size(); r++)
        {
            const vector<size_t> &round = rounds[r];
            {
                cout << "[panel] round " << r + 1 << "/" << rounds.size() << ": ";
                for (size_t j = 0; j < round.size(); j++)
                    cout << (j ? ", " : "") << members[round[j]].name;
                cout << endl;
            }
            auto t0 = chrono::steady_clock::now();
            bool roundGpu = false;
#ifdef FASTIBS_CUDA
            if (gpu && !gpuBroken)
            {
                bool resident = true;
                for (size_t i : round)
                    resident = resident && members[i].db->gpuActive();
                roundGpu = resident || uploadRound(round, nDev);
                if (!roundGpu)
                {
                    cerr << "[panel] device images unavailable; this and later rounds take the CPU path" << endl;
                    gpuBroken = true;
                }
            }
#endif
            if (!roundGpu)
                for (size_t i : round)
                    members[i].db->ensureRaOpen();
            const double setupSec = secondsSince(t0);
            setupTotal += setupSec;

            t0 = chrono::steady_clock::now();
            cout << "Calculating stats" << endl;
            vector<vector<Window>> stats(round.size(), vector<Window>(chunks.size()));
            thread_pool pool(KmerDatabase::poolThreads());
            boost::progress_display progressBar(chunks.size());
#ifdef FASTIBS_CUDA
            if (roundGpu)
                scanRoundGpu(round, chunks, stats, pool, progressBar);
            else
#endif
                scanRoundCpu(round, chunks, stats, pool, progressBar);
            cout << endl;
            const double statsSec = secondsSince(t0);
            statsTotal += statsSec;

            t0 = chrono::steady_clock::now();
            cout << "Writing stats to file" << endl;
            // A file that cannot be written fails the reference (thrown
            // below, once every other database's file is written).
            for (size_t j = 0; j < round.size(); j++)
                try
                {
                    KmerDatabase::writeStats(outPaths[round[j]], stats[j]);
                }
                catch (const std::exception &e)
                {
                    writeErrors += (writeErrors.empty() ? "" : "; ") + string(e.what());
                }
            const double writeSec = secondsSince(t0);
            writeTotal += writeSec;

            // Make room for the next round; a single round stays resident
            // for the next reference, as the single-database scan does.
            if (rounds.size() > 1)
                for (size_t i : round)
                    release(i);
            cout << "[panel] round " << r + 1 << "/" << rounds.size() << ": " << round.size()
                 << " databases, setup " << setupSec << " s, stats " << statsSec
                 << " s, write " << writeSec << " s" << endl;
        }
        keptResident = rounds.size() == 1 ? rounds[0] : vector<size_t>{};
        // The single-database "Scan phases" line, summed over the rounds,
        // so the breakdown scripts read a panel run like any other.
        cout << "Scan phases: setup " << setupTotal << " s, read " << ref.readSec
             << " s, window " << ref.windowSec << " s, stats " << statsTotal
             << " s, write " << writeTotal << " s" << endl;
        const double total = secondsSince(tRef);
        cout << "[panel] reference done: " << active.size() << " databases in "
             << rounds.size() << (rounds.size() == 1 ? " round, " : " rounds, ") << total
             << " s, per database " << total / active.size() << " s" << endl;
        if (!writeErrors.empty())
            throw std::runtime_error(writeErrors);
    }

private:
    struct Member
    {
        string prefix, name;
        std::unique_ptr<KmerDatabase> db;
    };
    vector<Member> members;
    int residentCap = 0;
    bool gpuBroken = false;
    vector<size_t> keptResident; // the last reference's single round, still resident

    void release(size_t i)
    {
#ifdef FASTIBS_CUDA
        members[i].db->releaseGpuIndex();
#endif
        members[i].db->releaseRa();
    }

    void releaseAll()
    {
        for (size_t i = 0; i < members.size(); i++)
            release(i);
        keptResident.clear();
    }

    // Bytes a database's whole image takes on a device, from the header
    // and the .kmc_pre size alone (LUT + signature map + counter-stripped
    // records), so rounds are planned before anything is read in full.
    static size_t imageBytesOf(const Member &m)
    {
        const auto &info = m.db->KMCInfo;
        const uint32_t suffixBytes = (info.kmer_length - info.lut_prefix_length) / 4;
        const size_t rec = suffixBytes <= 4 ? 4 : 8;
        size_t pre = 0;
        try
        {
            pre = (size_t)std::filesystem::file_size(m.prefix + ".kmc_pre");
        }
        catch (const std::exception &)
        {
        }
        const size_t norm = info.signature_len ? ((size_t)1 << (2 * info.signature_len)) * 4 : 0;
        return pre + norm + (size_t)info.total_kmers * rec;
    }

    // Resident sets in order: the longest prefix of what is left whose
    // images fit beside the pipeline buffers on the least-free device
    // (each device holds 1/nDev of every image), capped by --resident and
    // the kernels' MAX_DBS. On the CPU only the cap applies (host RAM is
    // not probed). A database that fits nowhere forms a round of its own
    // and the upload reports the shortfall as it does for a single run.
    vector<vector<size_t>> planRounds(const vector<size_t> &active, bool gpu, int nDev,
                                      const vector<SeqWindow> &chunks) const
    {
        size_t cap = active.size();
        if (residentCap > 0)
            cap = std::min<size_t>(cap, (size_t)residentCap);
#ifdef FASTIBS_CUDA
        cap = std::min<size_t>(cap, fastibs::MAX_DBS);
        size_t freeMin = 0;
        bool known = false;
        size_t batchBases = 0, windowCap = 0;
        int depth = 1;
        bool direct = false;
        if (gpu)
        {
            vector<int> devices;
            for (int d = 0; d < nDev; d++)
            {
                devices.push_back(d);
                if (const size_t f = DeviceKmerIndex::freeMemory(d))
                {
                    freeMin = known ? std::min(freeMin, f) : f;
                    known = true;
                }
            }
            size_t maxWinBases = 0;
            for (const auto &c : chunks)
                maxWinBases = std::max(maxWinBases, c.seq.size());
            batchBases = std::max((size_t)KmerDatabase::gpuOptions().batchMB << 20, maxWinBases);
            windowCap = std::max<size_t>(4096, batchBases / 256);
            depth = std::max(1, KmerDatabase::gpuOptions().streams);
            direct = nDev > 1 && EncodeProbeEngine::peerAccessible(devices);
        }
        auto pipelineBytes = [&](size_t n)
        {
            return EncodeProbeEngine::estimateDeviceBytes(nDev, direct, batchBases, windowCap,
                                                          depth, false, (int)n);
        };
        const size_t margin = 256ull << 20;
#endif
        vector<vector<size_t>> rounds;
        vector<size_t> cur;
        size_t curBytes = 0;
        for (size_t i : active)
        {
            bool fits = cur.size() < cap;
#ifdef FASTIBS_CUDA
            if (fits && gpu && known)
            {
                const size_t share = imageBytesOf(members[i]) / (size_t)std::max(1, nDev);
                fits = curBytes + share + pipelineBytes(cur.size() + 1) + margin <= freeMin;
                if (fits)
                    curBytes += share;
                else if (cur.empty())
                {
                    fits = true; // alone in its round; the upload says what is short
                    curBytes += share;
                }
            }
#endif
            if (!fits)
            {
                rounds.push_back(cur);
                cur.clear();
                curBytes = 0;
#ifdef FASTIBS_CUDA
                if (gpu && known)
                    curBytes = imageBytesOf(members[i]) / (size_t)std::max(1, nDev);
#endif
            }
            cur.push_back(i);
        }
        if (!cur.empty())
            rounds.push_back(cur);
        return rounds;
    }

#ifdef FASTIBS_CUDA
    // Uploads the round's images: with several devices every database is
    // split with ONE joint cut (KmcRaIndex::jointShardPlan from the
    // per-prefix counts of their LUTs, read from the .kmc_pre files alone),
    // so the device-side owner function is the same for all of them; one
    // device holds whole images, uploaded exactly as a single run uploads
    // its own. False when any upload failed (that database fell back to the
    // CPU path).
    bool uploadRound(const vector<size_t> &round, int nDev)
    {
        vector<vector<KmcRaIndex::Shard>> plans;
        if (nDev > 1)
        {
            vector<vector<uint64_t>> counts;
            for (size_t i : round)
                counts.push_back(KmcRaIndex(members[i].prefix, true).perPrefixCounts());
            plans = KmcRaIndex::jointShardPlan(counts, nDev);
        }
        bool ok = true;
        for (size_t j = 0; j < round.size(); j++)
        {
            KmerDatabase &db = *members[round[j]].db;
            db.gpuInitAttempted = true;
            db.initGpuIndex(plans.empty() ? nullptr : &plans[j]);
            ok = ok && db.gpuActive();
        }
        if (!ok)
            for (size_t i : round)
                members[i].db->releaseGpuIndex();
        return ok;
    }

    // The streamed pipeline over the round's images; database j's windows
    // are filled from its hit mask and the shared validity mask exactly as
    // the single-database consumer fills them.
    void scanRoundGpu(const vector<size_t> &round, const vector<SeqWindow> &chunks,
                      vector<vector<Window>> &stats, thread_pool &pool,
                      boost::progress_display &progressBar)
    {
        const uint32_t k = kmerSize();
        vector<vector<const DeviceKmerIndex *>> bsPerDb(round.size());
        for (size_t j = 0; j < round.size(); j++)
            for (const auto &p : members[round[j]].db->deviceShards())
                bsPerDb[j].push_back(p.get());
        for (size_t i = 0; i < chunks.size(); i++)
        {
            for (size_t j = 0; j < round.size(); j++)
            {
                stats[j][i].id = *chunks[i].id;
                stats[j][i].start = chunks[i].start;
                stats[j][i].end = chunks[i].end;
            }
            if (chunks[i].seq.size() < k)
                ++progressBar;
        }
        static std::mutex progressMx;
        KmerDatabase::runEngineLoop(
            bsPerDb, {}, k, chunks, pool,
            [&stats, &progressBar, k](size_t winIdx, const EncodeProbeEngine::HostSlot &hs,
                                      const fastibs::WindowDesc &d)
            {
                for (uint32_t j = 0; j < hs.nDbs; j++)
                {
                    Window &w = stats[j][winIdx];
                    fastibs::BitStats bs;
                    fastibs::BitStats::run(hs.hits(j), hs.validWords, d.outWordOff, d.nPos,
                                           bs, [&](uint64_t gap)
                                           { w.addVariationOriginal((int)gap, (int)k); });
                    w.totalKmers = (int)bs.totalKmers;
                    w.observedKmers = (int)bs.observedKmers;
                }
                std::lock_guard<std::mutex> lk(progressMx);
                ++progressBar;
            });
    }
#endif

    // The CPU path: one codec pass per window, one probe loop per database.
    void scanRoundCpu(const vector<size_t> &round, const vector<SeqWindow> &chunks,
                      vector<vector<Window>> &stats, thread_pool &pool,
                      boost::progress_display &progressBar)
    {
        vector<KmerDatabase *> dbs;
        for (size_t i : round)
            dbs.push_back(members[i].db.get());
        static std::mutex progressMx;
        for (size_t i = 0; i < chunks.size(); i++)
        {
            pool.push_task([i, &dbs, &chunks, &stats, &progressBar]
                           {
                               vector<Window> out(dbs.size());
                               KmerDatabase::getStatsFromSequenceMulti(chunks[i], dbs, out.data());
                               for (size_t j = 0; j < dbs.size(); j++)
                                   stats[j][i] = std::move(out[j]);
                               std::lock_guard<std::mutex> lk(progressMx);
                               ++progressBar;
                           });
        }
        pool.wait_for_tasks();
    }
};
