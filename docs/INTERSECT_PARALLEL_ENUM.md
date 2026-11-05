# KDBIntersect: parallel enumeration of the smaller database

## Why

`KDBIntersect` counts the k-mers two KMC databases share. On the GPU path
the larger database is device-resident and the smaller one is enumerated
on the host straight from its on-disk random-access index
(`KmcRaIndex::forEachKmer`: walk the per-bin look-up tables bucket by
bucket, reconstruct each full k-mer from the bucket's prefix and its suffix
record) and probed in batches (`DeviceKmerIndex::probeCountHost`, or
`DeviceHashIndex::probeCountHost` with `FASTIBS_PROBE=hash`), only the
per-batch count coming back. The measurements (jobs 68919/68918,
"enumeration-bound") show that the 2×10⁹ device probes
of the self-intersection take ~1.5 s of a 17.6 s run: the rest is the
single-threaded host enumeration. The tool was host-bound, so a faster
probe structure could not show — the hash backend lost by its build cost
everywhere — and the paper could not test its own claim that faster probe
structures pay where probing dominates. This change makes the
enumeration multi-threaded while preserving the exact count, so the
intersection becomes probe-bound.

## Design (`src/gpu/ParallelEnum.hpp`, `src/KDBIntersect.cpp`)

**Units.** The buckets of the LUT (a (bin, prefix) pair each; `lut[j]` to
`min(lut[j+1], total)` are its records) partition the records, and along
the on-disk order the LUT is non-decreasing (bin after bin, prefix after
prefix within a bin — the layout `shardLayout` relies on). The walk is
cut into *units*: contiguous ranges of whole buckets holding about one
batch of records each, placed by a binary search over the LUT for the
record targets `B, 2B, 3B, …` (`planEnumUnits`). The cuts are forced
increasing and end at the last bucket, so the units partition the buckets
whatever the LUT holds; a non-monotone LUT could only unbalance them,
never lose or duplicate a record. A bucket larger than a unit is one
unit by itself.

**Producers.** `FASTIBS_THREADS` threads (default: the CPUs in the
process's affinity mask, i.e. the SLURM allocation, as for the index
repack — `KmcRaIndex::usableCpus`; clamped to 1..256; the calling thread
is one of them) take units from a shared atomic counter and walk them with
`KmcRaIndex::forEachKmerIn(jBegin, jEnd, fn)`, the range form of
`forEachKmer` (which now delegates to it), so a unit is reconstructed
exactly as the sequential walk reconstructs it. Each producer owns one
batch buffer per *sink* — a sink is a device shard; the k-mer is routed
to the shard holding it by the same prefix-range (or `hashShardOf`) test
as before — and hands a full buffer to the sink's bounded queue, taking a
recycled buffer back. Partial buffers are handed in when the producer
runs out of units.

**Consumers.** Per sink, consumer threads pop batches, call the probe and
add its count. On the GPU path there is one consumer per device:
`probeCountHost` is synchronous and serializes on the shard's staging
buffers (and sets the calling thread's device itself), so a second
consumer per device would only wait. On the CPU path the probe is
`KmcRaIndex::probe` on the larger database held in host RAM
(bit-exact with `CKMCFile::CheckKmer`, the oracle of the device
kernels), read-only, so there is one consumer per thread.

**Exactness.** Every bucket is in exactly one unit; a unit is walked as
the sequential walk walks it; every k-mer lands in exactly one batch;
every batch is probed once; the intersection is a sum, so the order in
which batches reach a probe (or which producer filled them) does not
matter. With `FASTIBS_THREADS=1` the single producer takes the units in
on-disk order and the walk is the sequential one of the original path,
only cut into batches at bucket boundaries instead of at exactly 32M
keys.

**Batches and memory.** The batch is 32M keys (256 MB) per device for a
single-threaded run, as before, divided by the thread count (each
producer fills its own batches) down to a floor of 2M keys (16 MB), and
`FASTIBS_INTERSECT_BATCH_M=N` overrides it. A sink's queue holds at most
`min(threads, 256 MiB / batch bytes)` batches, so the queued buffers
are bounded by 256 MB per sink at any thread count; on top of that come
the producers' filling buffers (one per sink per producer, `threads ×
sinks × batch`, ≤ 256 MB per sink by the batch rule above unless the
floor binds) and the consumers' in-flight batches (one per device on the
GPU path). Worst cases: 1 device, 256 threads: 2M-key batches, 4 GB
filling + 256 MB queued + 16 MB in flight; 4 shards, 256 threads: 16 GB
filling + 1 GB queued — `FASTIBS_THREADS=16` (16 × 4 × 16 MB = 1 GB
filling) is the sensible setting on such a node, and the default is the
affinity mask, not the node's thread count. Drained buffers are recycled,
so a buffer is page-faulted once.

**Failure.** An exception in a probe call (CUDA error), in the owner
function or in the enumeration aborts the queues, which wakes every
blocked producer and consumer; all threads are joined and the exception
is rethrown, so `gpuIntersect` falls back to the CPU path as before (and
the CPU path to the KMC-API oracle). A `std::system_error` from thread
creation is handled the same way (abort, join what started, rethrow), so
no joinable thread is left behind. After the run the driver's key count
is checked against the database's `totalKmers`; a mismatch is reported
on stderr and treated as a failure of that path. `tests/phase3_test.cpp`
exercises the failure paths.

**Paths.** `FASTIBS_LEGACY=1` keeps the original KMC-API stream-and-probe
(single-threaded, the correctness oracle). Otherwise: GPU when built with
CUDA and a device is present (`FASTIBS_GPU=0` forces the CPU), else the
CPU path above. Both share the producer/consumer driver, so the CPU-only
build and the local tests exercise the new code. stdout stays the count
alone; one `[gpu]`/`[cpu] intersect:` line on stderr says how many k-mers
were enumerated by how many threads in how long ("enumerated in N s"
runs until the last producer finished and includes the time producers
spent waiting on full queues, so it is an upper bound on the walk
itself), how many batches, the wall time of enumeration + probing, and
the seconds spent inside the probe calls as a share of that wall time
(summed over consumers, so it can exceed 100 % with several devices or
on the CPU path) — near 100 % on one GPU means probe-bound. `cerr`'s
formatting flags are restored after the line.

## Environment variables

| Variable | Effect on `KDBIntersect` |
|---|---|
| `FASTIBS_THREADS=N` | Enumeration threads (default: the CPUs in the affinity mask, i.e. the SLURM allocation; clamped to 256; `1` = sequential walk). The measurement scripts set it explicitly for the labelled rows. |
| `FASTIBS_INTERSECT_BATCH_M=N` | Millions of keys per probe batch per device (default: 32 / threads, floor 2). |
| `FASTIBS_LEGACY=1` | Original KMC-API stream-and-probe (oracle). |
| `FASTIBS_GPU=0`, `FASTIBS_GPUS=N`, `FASTIBS_PROBE=hash`, `FASTIBS_HASH_LF=F` | Unchanged. |

## Gate (CPU-only build, WSL Ubuntu, g++ 15.2, 8 hardware threads)

Corpora from `scripts/make_test_corpus.sh` (k, reference bp, seed):
`s1` (25, 2M, 11), `s2` (25, 2M, 12), `mid` (25, 100M, 13), `k31`
(31, 20M, 14); `s1ref`/`midref` are KMC databases of the s1/mid
*reference* FASTA (the sample databases are mutated copies, so these
cross-intersections are large but not total; random genomes with
different seeds share no 25-mers, as the oracle confirms). For each
pair the counts of `FASTIBS_THREADS=1` (sequential walk), `8` and `3`,
of `FASTIBS_LEGACY=1`, of the pre-change binary, and of the independent
oracle `kmc_tools simple a b intersect` → `kmc_tools info` must agree.

| pair (enumerated × probed) | threads 1 | threads 8 | threads 3 | `FASTIBS_LEGACY=1` | pre-change binary | `kmc_tools` | equal |
|---|---|---|---|---|---|---|---|
| s1 × s1 | 2,202,805 | 2,202,805 | 2,202,805 | 2,202,805 | 2,202,805 | 2,202,805 | yes |
| s1 × s2 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| s2 × s1 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| s1 × s1ref | 1,821,136 | 1,821,136 | 1,821,136 | 1,821,136 | 1,821,136 | 1,821,136 | yes |
| s1ref × s1 | 1,821,136 | 1,821,136 | 1,821,136 | 1,821,136 | 1,821,136 | 1,821,136 | yes |
| mid × mid | 100,202,793 | 100,202,793 | 100,202,793 | 100,202,793 | 100,202,793 | 100,202,793 | yes |
| mid × s1 | 1 | 1 | 1 | 1 | 1 | 1 | yes |
| mid × midref | 83,023,327 | 83,023,327 | 83,023,327 | 83,023,327 | 83,023,327 | 83,023,327 | yes |
| midref × mid | 83,023,327 | 83,023,327 | 83,023,327 | 83,023,327 | 83,023,327 | 83,023,327 | yes |
| k31 × k31 | 20,202,781 | 20,202,781 | 20,202,781 | 20,202,781 | 20,202,781 | 20,202,781 | yes |
| mid × KMC1(mid), both orders | 100,202,793 | 100,202,793 | — | — | — | = mid total | yes |

(The tool enumerated the database with the smaller `.kmc_pre` when this
table was made; it now enumerates the one with fewer k-mers, using the
`.kmc_pre` size only to break ties. Either way "both orders" of a pair
exercise the same assignment; they are listed because the command line
differs. All runs: CPU path of the CPU-only build,
2026-10-27.)

The bonus row intersects `mid` with the `kmc_tools` output of its own
self-intersection, which is a KMC1-format database (one LUT, no
signature bins), so the single-bin walk is covered too.
`tests/phase0..4` (built with their header `g++` lines) pass;
`phase3_test` gained a section that checks the parallel driver delivers
exactly the sequential k-mer set to the probe batches for producer
counts {1, 2, 5} × sinks {1, 3} × batch sizes {7, 1000, 4M}, routed to
the right sink; that a failing probe and a failing owner function each
stop every thread and surface; that a sink which never receives a batch
finishes cleanly with the count whole; and that slow consumers (a 5 ms
sleep per 50k-key batch, so the producers block on the bounded queue)
still deliver the set. It is run on `s1` and `k31` (KMC2) and on the
KMC1-format `kmc_tools` copy of `s1` (`s1_kmc1`).

## Local timing (indicative; CPU path, mid self-intersection, 3 runs)

`mid` × `mid` (100,202,793 k-mers, count 100,202,793), CPU-only build,
WSL on a laptop (8 hardware threads), seconds, medians of 3 [min–max]:

| | process wall | enumeration + probe wall (`[cpu]` line) | inside the probe calls |
|---|---|---|---|
| `FASTIBS_THREADS=1` | 8.97 [8.76–9.02] | 8.42 [8.25–8.45] | 8.19 [8.12–8.30] (1 consumer) |
| `FASTIBS_THREADS=8` | 2.39 [2.34–2.49] | 1.87 [1.83–1.97] | 14.27 [14.00–14.94] summed over 8 consumers |
| `FASTIBS_LEGACY=1` (KMC API, n = 1) | 17.40 (30.17 in an earlier, colder run) | — | — |

1 → 8 threads: 3.75× on the process wall (the two index loads, ~0.5 s,
stay serial), 4.5× on enumeration + probe; 7.3× against the KMC-API
path. On the CPU the cost is the host probe (12 Mprobe/s per thread
here), not the enumeration (55 Mkeys/s on one thread on `s1`; the
"enumerated in N s" figure of the `[cpu]` line includes the time a
producer waits on a full queue), so these numbers show that the driver
and its consumers scale; the enumeration speed-up itself can only be
measured where the probe is fast, i.e. on the GPU. (Numbers from the
reviewed build: byte-budgeted queue, affinity-mask default, self-check.)

## What to measure on the cluster

One A100 (the paper's platform), `corpus_big` self-intersection (2×10⁹
k-mers, count 2,000,199,169), `FASTIBS_THREADS=1` vs `16` (set it
explicitly: the default is the node's hardware concurrency, 128–256, not
the 16-CPU allocation), medians of 3:

1. `bin/KDBIntersect $DB $DB` — BS backend. Expect the 17.6 s of job
   68919 at 1 thread and, at 16 threads, the enumeration to drop under
   the probe side, on top of the two index loads and the upload. The
   `[gpu] intersect:` line gives the enumeration+probe wall and the share
   of it inside the probe calls: near 100 % at 16 threads means
   probe-bound. **The floor is not the kernel** (2×10⁹ probes at
   1.3 Gq/s ≈ 1.5 s) but `probeCountHost`'s synchronous staging path:
   per call one pageable `cudaMemcpy` of the batch (16 MB at 2M keys; 16
   GB in all, ~2–3 s at pageable-copy rates, during which the device
   idles), the kernel, then a `cudaMalloc`/`cudaFree` pair each for the
   CUB reduction's output and temp storage plus the 8-byte D2H — about
   1000 calls at 2M-key batches, 62 at 32M. Expect roughly 3–5 s of
   probe calls per 2×10⁹ keys on one A100, i.e. the run becomes
   probe-bound on the *staging*, not on the search. The next step, once
   the numbers confirm this, is in `DeviceKmerIndex`/`DeviceHashIndex`
   (not touched here): pinned staging buffers with `cudaMemcpyAsync` on a
   stream, double-buffered so the copy of batch *i+1* overlaps the kernel
   of batch *i*, and the reduction scratch allocated once per shard.
2. `FASTIBS_PROBE=hash bin/KDBIntersect $DB $DB` — hash backend, same
   two thread counts. Once probe-bound, the hash set's higher probe
   throughput (4.9× at the kernel level) can show against its 4.5 s
   build only to the extent the kernel, not the staging, is the floor;
   whether it repays the build at this size is the paper's open
   question.
3. `FASTIBS_INTERSECT_BATCH_M=8` and `=32` at 16 threads, BS and hash,
   to size the per-call fixed cost above against the copy volume
   (`intersect_hash.sbatch`'s `bs-b8`/`bs-b32`/`hash-b8`/`hash-b32`
   rows).
4. CPU baseline: `FASTIBS_GPU=0 FASTIBS_THREADS=16` is now the parallel
   CPU path (expect far below the 293 s of job 68881, which was the
   single-threaded KMC-API path); `FASTIBS_LEGACY=1` reproduces that
   original number. The paper's intersection speed-up has to say which
   CPU path it is against.

## Risks

- The GPU path was not compiled or run here (no CUDA toolkit in WSL).
  The `.cu` files are untouched and the only contract used is the
  unchanged `probeCountHost(const std::vector<uint64_t>&)`; the CUDA
  block of `KDBIntersect.cpp` was type-checked with `g++ -fsyntax-only
  -DFASTIBS_CUDA` against the real `.cuh` declarations (a one-line stub
  `cuda_runtime.h`). The consumer thread per device is a new thread, as
  the previous `std::async` helpers were; `probeCountHost` sets its own
  device.
- Default thread count is the affinity mask (the SLURM allocation),
  clamped to 256 — not the scan pool's hardware-concurrency default, so
  the two tools differ in what an unset `FASTIBS_THREADS` means; a
  whole-node run without a cpuset still gets every hardware thread as a
  producer (bounded memory, see above, but oversubscribed).
- Batch boundaries moved from exactly 32M keys to bucket boundaries, so
  the sequence of `probeCountHost` calls differs from the previous
  binary's even at 1 thread (the count does not), and at 16 threads
  there are ~16× more, smaller calls on the synchronous staging path
  described under item 1 above.
- The CPU path now loads both databases through `KmcRaIndex` (4–8 bytes
  per k-mer each) instead of the KMC API's random-access view of the
  larger one (which also held the counters): comparable host RAM, and
  it changes the CPU intersection baseline (see item 4 above).
