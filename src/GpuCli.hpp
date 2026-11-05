#pragma once

// Shared CLI parsing for the GPU/pipeline knobs, used by fastibs and
// fastibsmapper (both drive the streamed pipeline) so the flags mean the
// same thing in both. KDBIntersect keeps its original two-argument interface
// and follows the FASTIBS_GPU / FASTIBS_LEGACY environment switches only.

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "CliInput.hpp"
#include "KmerDatabase.hpp"

// The option block shared by the tools' usage texts.
inline const char *gpuOptionsUsage()
{
    return "Options (GPU builds):\n"
           "  --gpu            Force the GPU path (default: auto when a device is present)\n"
           "  --cpu            Force the CPU path\n"
           "  --batch-mb=N     Sequence megabytes per GPU batch (default 128; halved\n"
           "                   automatically until the pipeline buffers fit beside\n"
           "                   the index on the device)\n"
           "  --streams=N      GPU pipeline depth in streams (default 3)\n"
           "  --sync           Per-window synchronous GPU probing (Phase 1 path)\n"
           "  --sort-queries   Sort each batch by k-mer value before probing (GPU-*-SORTED; one device)\n"
           "  --probe=bs|hash  Probe backend: ported KMC index (default) or hash set\n"
           "  --probe=batched  CPU option, in every build (implies --cpu): K probes in\n"
           "                   flight per thread over the host index image\n"
           "                   (CPU-BATCHED); K = FASTIBS_CPU_BATCH (default 16)\n"
           "  --hash-lf=F      Hash-set load factor (default 0.6)\n"
           "  --pin-lut        Keep the prefix LUT in persisting L2 cache\n"
           "  --gpus=N|all     Devices to use (default: all visible, or FASTIBS_GPUS=N).\n"
           "                   The index is split across them: each device holds a\n"
           "                   k-mer prefix range (~1/N of the index); a batch goes to\n"
           "                   one device, which encodes it and routes the queries to\n"
           "                   their devices over NVLink / PCIe peer-to-peer; output is\n"
           "                   identical for any N\n"
           "  --no-peer        Route the queries through host-staged copies even where\n"
           "                   the devices have peer access (as FASTIBS_NO_PEER=1)\n"
           "  -h, --help       Print this text and exit\n"
           "  --version        Print the GPU-IBS version and build, and exit\n";
}

// Panel mode (fastibs only; docs/MULTIDB_SCAN.md): more databases scanned
// in the same reference pass as <sourcePath>.
struct PanelOptions
{
    std::vector<std::string> dbs; // --db=<path>, in order
    std::string listFile;         // --db-list=<file>
    int resident = 0;             // --resident=N (0: as many as fit)
};
inline PanelOptions &panelOptions()
{
    static PanelOptions o;
    return o;
}

inline const char *panelOptionsUsage()
{
    return "Panel mode (several databases, one reference pass):\n"
           "  --db=<path>      Another KMC dataset folder to scan in the same pass\n"
           "                   (repeatable); its output file is named as in a\n"
           "                   single run. The reference is read and packed once,\n"
           "                   every batch is encoded once and probed against every\n"
           "                   resident database; outputs are identical to N runs\n"
           "  --db-list=<file> A file with one dataset folder per line (# comments)\n"
           "  --resident=N     At most N databases resident at once (default: as\n"
           "                   many as fit the device(s); the rest in further rounds)\n";
}

// Parses argv[firstOpt..argc-1] into KmerDatabase::gpuOptions() (and, when
// allowPanel, panelOptions()). Returns false (after reporting the offending
// argument) on an unknown option or a bad value.
inline bool parseGpuOptions(int argc, char *argv[], int firstOpt, bool allowPanel = false)
{
    using std::cerr;
    using std::cout;
    using std::endl;
    using std::string;

    auto &gpuOpts = KmerDatabase::gpuOptions();
    // --probe=batched implies --cpu; a later --probe=bs|hash undoes that
    // implication (restoring what --gpu/--cpu/auto said before it) rather
    // than leaving a silent CPU-OPT run.
    int enableBeforeBatched = gpuOpts.enable;
    // A whole integer of at least `lo` (no trailing characters: "12x" is
    // an error, not 12).
    auto parseIntOption = [](const string &arg, size_t prefixLen, int &out, int lo = 1)
    {
        if (cli::parseInt(arg.substr(prefixLen), lo, out))
            return true;
        cerr << "Invalid value for option: " << arg << " (expected an integer >= " << lo << ")"
             << endl;
        return false;
    };
    for (int a = firstOpt; a < argc; a++)
    {
        const string arg = argv[a];
        if (arg == "--gpu")
            gpuOpts.enable = 1;
        else if (arg == "--cpu")
            gpuOpts.enable = 0;
        else if (arg == "--sync")
            gpuOpts.sync = true;
        else if (arg.rfind("--batch-mb=", 0) == 0)
        {
            if (!parseIntOption(arg, 11, gpuOpts.batchMB))
                return false;
        }
        else if (arg.rfind("--streams=", 0) == 0)
        {
            if (!parseIntOption(arg, 10, gpuOpts.streams))
                return false;
        }
        else if (arg == "--gpus=all")
            gpuOpts.gpus = 0; // every visible device (the default)
        else if (arg == "--gpus=0")
        {
            // FASTIBS_GPUS=0 means every device, a --gpus=0 could as well
            // mean none: neither is guessed.
            cerr << "Invalid value for option: --gpus=0 (use --gpus=all for every device, "
                    "--cpu for none)"
                 << endl;
            return false;
        }
        else if (arg.rfind("--gpus=", 0) == 0)
        {
            if (!parseIntOption(arg, 7, gpuOpts.gpus))
                return false;
        }
        else if (arg == "--no-peer")
            gpuOpts.noPeer = true;
        else if (allowPanel && arg.rfind("--db=", 0) == 0 && arg.size() > 5)
            panelOptions().dbs.push_back(arg.substr(5));
        else if (allowPanel && arg.rfind("--db-list=", 0) == 0 && arg.size() > 10)
            panelOptions().listFile = arg.substr(10);
        else if (allowPanel && arg.rfind("--resident=", 0) == 0)
        {
            // 0 is the default (as many as fit)
            if (!parseIntOption(arg, 11, panelOptions().resident, 0))
                return false;
        }
        else if (arg == "--sort-queries")
            gpuOpts.sortQueries = true;
        else if (arg == "--probe=bs" || arg == "--probe=hash")
        {
            gpuOpts.probeHash = arg == "--probe=hash";
            if (gpuOpts.cpuBatched)
            {
                gpuOpts.cpuBatched = false;
                gpuOpts.enable = enableBeforeBatched;
            }
        }
        else if (arg == "--probe=batched")
        {
            // A CPU probe backend, so it also selects the CPU path (the
            // last of --gpu/--cpu/--probe=batched wins, as between
            // --gpu and --cpu).
            if (!gpuOpts.cpuBatched)
                enableBeforeBatched = gpuOpts.enable;
            gpuOpts.cpuBatched = true;
            gpuOpts.enable = 0;
        }
        else if (arg == "--pin-lut")
            gpuOpts.pinLut = true;
        else if (arg.rfind("--hash-lf=", 0) == 0)
        {
            gpuOpts.probeHash = true; // the load factor only exists for hash
            try
            {
                size_t used = 0;
                gpuOpts.hashLoadFactor = std::stod(arg.substr(10), &used);
                if (used != arg.size() - 10) // trailing characters
                    gpuOpts.hashLoadFactor = 0;
            }
            catch (const std::exception &)
            {
                gpuOpts.hashLoadFactor = 0;
            }
            if (gpuOpts.hashLoadFactor < 0.1 || gpuOpts.hashLoadFactor > 0.95)
            {
                cerr << "Invalid value for option: " << arg
                     << " (expected 0.1 .. 0.95)" << endl;
                return false;
            }
        }
        else
        {
            cerr << "Unknown option: " << arg << endl;
            return false;
        }
    }
#ifdef FASTIBS_CUDA
    if (gpuOpts.noPeer)
        EncodeProbeEngine::forceStaged();
#else
    if (gpuOpts.enable == 1)
        cerr << "Warning: built without CUDA (-DFASTIBS_CUDA=ON); --gpu has no effect, "
                "running on the CPU"
             << endl;
#endif
    return true;
}
