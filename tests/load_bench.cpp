// Times the on-disk index read + repack (KmcRaIndex construction) — the
// "loads" component of every GPU run — for a thread-count sweep, and
// doubles as the cold-vs-warm I/O diagnostic for a cluster filesystem.
//
// Build (from repo root):
//   g++ -O2 -std=c++2a tests/load_bench.cpp -lpthread -o load_bench
// Run:
//   ./load_bench <kmc_db_path_prefix> [threads ...]   (default sweep 1 2 4 8 16)
// A thread count of 0 means the loader's own default — the CPUs in the
// process's affinity mask (the SLURM allocation, not the whole node); the
// library gives FASTIBS_LOAD_THREADS=0 the same meaning.
//
// Reading the same database twice in a row (e.g. "0 0") separates I/O from
// CPU: a first run far slower than the second was a cold read from storage;
// two equal runs at GB/s came from the page cache; two equal slow runs mean
// the filesystem is not caching at all. Warm the page cache first
// (cat <db>.kmc_suf > /dev/null) when only the repack itself is of interest.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../src/gpu/KmcRaIndex.hpp"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: %s <kmc_db_path_prefix> [threads ...]   (0 = loader default)\n",
               argv[0]);
        return 1;
    }
    std::vector<int> sweep;
    for (int i = 2; i < argc; i++)
        sweep.push_back(atoi(argv[i]));
    if (sweep.empty())
        sweep = {1, 2, 4, 8, 16};

    for (int t : sweep)
    {
        setenv("FASTIBS_LOAD_THREADS", std::to_string(t).c_str(), 1);
        KmcRaIndex index(argv[1]);
        const double s = index.loadSeconds();
        const double gb = index.loadBytes() / 1e9;
        printf("threads=%2u  read+repack %.3f s  (%llu records, %.1f Mrec/s, "
               "%.1f GB at %.2f GB/s)\n",
               index.loadThreads(), s,
               (unsigned long long)index.getParams().totalKmers,
               index.getParams().totalKmers / s / 1e6, gb, gb / s);
    }
    return 0;
}
