# CPU-BATCHED: the K-way software-pipelined CPU probe

A stronger CPU baseline for the paper's "GPU vs. a whole socket" comparison.
The question a reviewer will ask: how much of the GPU's advantage over a
128-core socket survives a CPU path tuned as hard as the GPU path? CPU-OPT
(the default CPU path) probes one canonical k-mer at a time per thread
through the KMC API, so every step of the binary search stalls on one DRAM
miss and a core keeps a single miss in flight; the GPU wins mainly by having
thousands of probes in flight. CPU-BATCHED is the CPU counterpart: K probes
in flight per worker thread.

## What it does

Per worker thread (one window task of the scan, one chunk of the mapper), the
canonical codes come from the same rolling codec as CPU-OPT (`KmerCodec`).
Instead of `CKMCFile::IsKmer` per code, `fastibs::probeBatched`
(`src/CpuProbeBatched.hpp`) takes the codes K at a time as **two independent
groups of K/2** (`ProbeGroup`; K = 1 keeps the second group empty and is the
plain probe with prefetches) and runs their membership probes interleaved
**level by level**, each level prefetched (`__builtin_prefetch`) for a whole
group before any of it is read:

1. prefix routing and the group's minimizer signatures (`signatureOf`, the
   shared host/device routine), prefetch of its signature-map entries;
2. the bin bases from the signature map, prefetch of the pairs of LUT
   entries (the bucket's two bounds);
3. the bucket bounds and compare targets (empty buckets drop out here);
4. the binary searches, software-pipelined across the two groups: A's next
   records are prefetched, then B's current records are read and stepped,
   then B's next records are prefetched, then A's are read and stepped, and
   so on — one group's lines are always in flight while the other group's
   step computes, so the memory system never idles on compute. The step is
   branchless: the compare selects the new bound with conditional moves
   (`lo = lt ? mid+1 : lo; hi = lt ? hi : mid-1; hit = eq`), so a
   mispredicted hit/miss cannot drain the loads in flight; a slot leaves the
   group when it hits or its range empties, and the compaction pass runs
   only when some slot did. Levels 1-3 alternate the groups the same way.

So K independent cache misses are outstanding per thread at every level,
instead of one. The search arithmetic is `fastibs::probeOne`
(`src/gpu/ProbeCore.hpp`, the template every CUDA kernel calls) step for
step, over the **host image** of the index (`KmcRaIndex`: the
counter-stripped, native-integer arrays the GPU path uploads — read straight
from the `.kmc_pre/.kmc_suf` files, not the vendored KMC random-access view),
so the answer is bit-identical to the GPU kernel's and to
`CKMCFile::IsKmer`. `tests/phase1_test.cpp` section 4 checks the batched
probe against `KmcRaIndex::probe` query for query (codec queries, random
k-mers, a validity mask, K = 1..512). Slot positions are `size_t`, so any
chunk length is fine.

Statistics are unchanged: the scan runs the same run-length gap accumulation
over the hit flags (as the GPU-sync host path does), the mapper the same
difference-array accumulation (`accumulateMappingChunk`, chunk-parallel).
The validity mask of the mapper's per-position encode keeps non-ACGT
positions out of the probe groups.

Memory: the host image holds 8 bytes per k-mer when the suffix exceeds 4
bytes (the KMC view holds suffix + counter bytes: 6 per k-mer for the
production database), i.e. 16 GB instead of 12 GB for the 2×10⁹-k-mer
corpus; only one of the two is loaded. The load prints
`[cpu] index read+repack: ... s (N threads, G GB, GB/s)`, the same line the GPU
path prints with a `[gpu]` tag (same loader: `FASTIBS_LOAD_THREADS`, default
the CPUs of the affinity mask). A load failure (missing/truncated/unsupported
file) exits with `Error: Could not open KMC database for the batched probe:
<reason>`, as a failed KMC open does.

## The option

- `--probe=batched` on `fastibs` and `fastibsmapper` (next to the existing
  `--probe=bs|hash` of the GPU path; a CPU option, available in every build,
  the usage text says so). It is a CPU probe backend, so it implies `--cpu`;
  a later `--probe=bs|hash` undoes that implication (restores what
  `--gpu`/`--cpu`/auto said before it) instead of leaving a silent CPU-OPT
  run; a later `--gpu` wins as it does over `--cpu`. Any unknown option still
  prints the usage.
- `FASTIBS_CPU_PROBE=batched` selects it from the environment **when the CPU
  path probes** (it does not by itself force the CPU path; `FASTIBS_GPU=0` or
  `--cpu` does). Any other non-empty value warns once on stderr and keeps
  CPU-OPT.
- `FASTIBS_LEGACY=1` (CPU-BASE, the string-based oracle) takes precedence and
  warns once on stderr that `--probe=batched` / `FASTIBS_CPU_PROBE=batched`
  is ignored.
- `FASTIBS_CPU_BATCH=K`: probes in flight per thread, default 16, clamped to
  1..256 (`fastibs::kCpuBatchMax`); split K/2 + K/2 over the two groups.
- `FASTIBS_THREADS=N` sizes the thread pool as before (default: the node's
  hardware concurrency); the pool honours it on both tools, which is what the
  full-socket script relies on.
- The log line says which path ran:
  `K-mer path: rolling codec + 16-way pipelined probe of the host index image (CPU-BATCHED)`.
- Untouched: CPU-OPT (default) and CPU-BASE run exactly the code they ran
  before; a GPU run ignores the knob; `KDBIntersect` has no options and is
  not affected.

## Gate (local, WSL Ubuntu, g++ 15.2 `-O3`, CPU-only build, `FASTIBS_THREADS=8`)

Corpora from `scripts/make_test_corpus.sh` (`<dir> k ref_bp seed`):
`small` = 25 2000000 1 (2,202,805 k-mers, LUT prefix 5, signature 9, 64-bit
records); `k31` = 31 20000000 5 (20,202,781 k-mers, LUT prefix 3, 64-bit);
`mid` = 25 100000000 3 (100,202,787 k-mers, LUT prefix 5, 64-bit);
`k21` = 21 2000000 7 (2,202,819 k-mers, suffix 16 symbols: the 32-bit-record
instantiation). Every row: CPU-OPT (`--cpu`) vs CPU-BATCHED
(`--cpu --probe=batched`, `FASTIBS_CPU_BATCH=K`), `cmp` of the output files,
after the review's changes (branchless step, two groups).

| corpus | tool | W | K = 1 / 8 / 16 / 32 | bytes |
|---|---|---|---|---|
| small | fastibs | 50000 | identical / identical / identical / identical | 2,062 |
| small | fastibs | 10000 | identical / identical / identical / identical | 8,505 |
| small | fastibsmapper | — | identical / identical / identical / identical | 6,316,590 |
| k21 | fastibs | 50000 | identical / identical / identical / identical | 2,029 |
| k21 | fastibsmapper | — | identical / identical / identical / identical | 6,322,491 |
| k31 | fastibs | 50000 | identical / identical / identical / identical | 17,590 |
| k31 | fastibsmapper | — | identical / identical / identical / identical | 57,733,546 |
| mid | fastibs | 50000 | identical / identical / identical / identical | 87,686 |
| mid | fastibsmapper | — | identical / identical / identical / identical | 286,716,446 |

Switch checks on `small` (W = 50000, all outputs identical to CPU-OPT's 2,062
bytes): `FASTIBS_CPU_PROBE=batched` takes the batched path;
`FASTIBS_CPU_PROBE=bogus` warns once and runs CPU-OPT; `FASTIBS_LEGACY=1
--probe=batched` warns once and runs CPU-BASE; `--probe=batched --probe=bs`
runs CPU-OPT. The 15 timed runs below are identical to the gate reference as
well. The pre-review (lock-step, single-group) version passed the same gate
at K = 1/16/64.

Regression tests (`tests/`, built with the `g++` lines in their headers):
phase0, phase1 (with the batched-probe section), phase2, phase3, phase4 all
pass on the small corpus; phase1 also passes on the k21 (32-bit records) and
k31 (64-bit records) corpora with the reviewed probe.

## Local timings (indicative only: a laptop under WSL, 8 threads, noisy)

`mid` corpus, W = 50000, `FASTIBS_THREADS=8`, 3 runs each rep-major, the
tool's own `Reference processed in` seconds, the `stats` phase in
parentheses. `v1` = the pre-review probe (lock-step, one group of K), `v2` =
the reviewed probe (branchless step, two groups of K/2):

| configuration | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| CPU-OPT | 21.80 (21.02) | 18.33 (17.83) | 18.69 (18.20) | **18.69** (18.20) |
| v1, K=16 | 13.63 (12.82) | 10.40 (9.72) | 10.10 (9.44) | **10.40** (9.72) |
| v2, K=16 | 11.11 (10.38) | 15.32 (14.46) | 10.85 (10.21) | **11.11** (10.38) |
| v2, K=8 | 11.03 (10.46) | 13.05 (12.39) | 13.79 (13.07) | **13.04** (12.39) |
| v2, K=32 | 13.62 (13.11) | 10.32 (9.61) | 10.24 (9.64) | **10.32** (9.64) |

About 1.8× on the probe phase here, with a 0.8 GB index that a laptop's L3
partly covers and a memory system far from a socket's; the two-group change
is within the run-to-run noise locally (v1 and v2 at K=16 overlap; the
machine's best runs are 9.4-9.7 s for v1 and 9.6-10.2 s for v2, with one
15 s outlier each way). K = 8 trails, K = 32 ≈ K = 16. Mapper, one run each
(reviewed probe): CPU-OPT probe phase 15.46 s, K=8 10.05 s, K=16 11.02 s,
K=32 9.85 s. These numbers must not appear in the paper; the cluster row is
what counts, and it is where the two-group pipelining and the branchless
step are expected to matter (16 GB index, every record a DRAM miss).

## What to measure on the cluster

The whole-node CPU job (one EPYC 9754, the `corpus_big` 2×10⁹-k-mer
database) runs, in the same reps and with the
same `taskset` pinning and `FASTIBS_THREADS` as the CPU-OPT rows (whose
command lines are unchanged):

- scan: CPU-BATCHED K=16 at 16c, 128c and 256t; K=8 and K=32 at 128c and
  256t (`cpu-bat16-16c`, `cpu-bat8-128c`, `cpu-bat16-128c`, `cpu-bat32-128c`,
  `cpu-bat8-256t`, `cpu-bat16-256t`, `cpu-bat32-256t`), x3, every output
  `cmp`-ed against the first pass;
- mapper: CPU-BATCHED K=16 at 128c and 256t, x1;
- every row records, besides wall and `Reference processed in` (proc), the
  **probe phase alone** (`phase_s`: the scan's `stats` phase from its `Scan
  phases` line, the mapper's `probe` phase), and the summary gives its median
  next to proc. proc includes the index load — KMC's single-threaded open
  for CPU-OPT vs. the multi-threaded repack for CPU-BATCHED — and the paper
  must compare the probe phase, not a setup advantage;
- the platform record prints the THP mode
  (`/sys/kernel/mm/transparent_hugepage/enabled` and `defrag`) and
  `lscpu | grep NUMA`, and for every batched configuration's first rep the
  `Rss`/`AnonHugePages` lines of the running process's `smaps_rollup`, read
  once its host image is loaded — whether the kernel backed the 16 GB suffix
  array with huge pages.

The paper's row is the full-socket scan: GPU (A100, 9.36 s) vs. CPU-OPT at
128c/256t vs. CPU-BATCHED at 128c/256t, medians of the three passes, probe
phase and whole scan both. What to expect: with K misses in flight per core
the socket moves from latency-bound to bandwidth-bound — about 9 distinct
cache lines per probe (the ~log₂(bucket) first steps of the search, bucket
≈ 4,000 records of 8 bytes) for 2×10⁹ probes is ~1.2 TB of random 64-byte
reads, so the floor at the socket's DDR5 bandwidth is a few seconds and the
achieved figure depends on how well the random-access stream is sustained.
The expected outcome is a whole-scan gap to the A100 of roughly 1.5-2.3×
(down from 2.6× against CPU-OPT), with a larger gap on the probe phase alone
(the scan's read/window/write phases are the same on both). If K=32 is no
better than K=16 on the socket, the miss queue, not K, is the limit (a Zen 4
core has a few dozen outstanding L1 misses) and the K=8 row shows where the
knee is.

## What a reviewer will still ask

- **Huge pages.** Random 8-byte reads over a 16 GB array miss the TLB on
  nearly every access with 4 KB pages (a page walk per probe step on top of
  the DRAM miss). The THP mode of the node (`always` backs large anonymous
  mappings automatically; `madvise` needs the application to ask) and the
  `AnonHugePages` sample in the job log say whether the batched runs had
  them. The next experiment is a 2 MB-aligned allocation of the suffix array
  with `madvise(MADV_HUGEPAGE)` in `KmcRaIndex` (and 1 GB pages via
  `hugetlbfs` if the node reserves any) — not done here because that file
  was being reworked at the same time; it would also serve the GPU path's
  host staging.
- **Per-core K under SMT.** With both hardware threads of a core running a
  worker (the 256t rows), two threads share one core's miss queue, so the
  effective K per core is 2K; the 128c vs 256t rows at K=8/16/32 separate
  "more threads" from "more probes per thread". The fair statement is probes
  in flight per core, not per thread.
- **Which number is the baseline.** CPU-BATCHED is the same algorithm as the
  GPU's probe (same index image, same search), so the comparison is
  hardware-to-hardware; the expected outcome is a whole-scan gap to the A100
  of roughly 1.5-2.3× with a larger gap on the probe phase alone, and the
  paper should report both and say that the CPU path's remaining phases
  (reading the FASTA, windowing, writing) are shared with the GPU path.
- **Bandwidth accounting.** Report the achieved random-read bandwidth
  (lines touched per probe × probes / probe-phase seconds) against the
  socket's DDR5 peak, so the reader can see how close the batched CPU path is
  to its memory floor; the GPU's advantage then reads as HBM bandwidth and
  TLB reach, not as better code.
