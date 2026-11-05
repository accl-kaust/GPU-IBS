# One reference pass, N databases (panel mode)

Design note for the multi-database scan of `fastibs`: the reference is read,
windowed and 2-bit packed **once**, N sample databases are resident on the
GPU(s) at the same time, every packed batch is encoded once and probed against
every resident database, and the per-window statistics and the per-sample TSVs
come out for all N, byte-identical to N separate runs.

## 1. Why

A measured breakdown of where a scan's time goes (job 68921) is the
motivation. One A100, a 2 Gbp reference, a
2×10⁹-k-mer database, window 50 kb:

| phase (one sample, one reference) | wall s | nature |
|---|---|---|
| index read + repack + upload | 4.2 | per database |
| FASTA read | 4.1 | per reference |
| streamed pipeline (pack + probe + statistics) | 2.0 | pack 20 thread-s, statistics 15 thread-s on 16 threads; the device was waited for 1.0 s |
| write | 0.1 | per (database, reference) |
| **total** | **10.5** | |

Once the membership probe runs on the device it is a fifth of the scan; the
host sets the pace. The real workload is a *panel*: hundreds of sample
databases scanned against the same reference(s). Today each sample is a
separate process that re-reads the reference (4.1 s), re-packs it (20
thread-s) and re-runs the host pipeline. Panel mode makes the reference-side
work a one-off and leaves per sample only what is genuinely per sample: the
database's own image and its own statistics and output.

## 2. Overview

```
fastibs <db1> <refs> <results> <W> --db=<db2> --db=<db3> ...      (or --db-list=<file>)

for each reference .fasta:                    read + window ONCE
    for each ROUND (a resident set of databases that fits the device(s)):
        upload the round's images            (per database; the only per-sample setup)
        stream the reference once:
            pack batch -> H2D -> K1 once, K2 x n -> D2H (n hit masks + 1 validity mask)
            per window, per database: run-length statistics      (per sample)
        write one TSV per database           (per sample)
        free the round's images unless it is the only round
```

Everything the single-database run does is still done, in the same order and
by the same code; what changes is *how many times*. The outputs are the same
files with the same names (`<database>_v_<ref>_<W>.tsv`), produced by the same
writer from the same `Window` records, so byte-identity holds by construction
once the per-database hit bits are the same, which the shared host/device
templates guarantee (section 7).

## 3. Data layout on the device

**One device.** Database i is one `DeviceKmerIndex` image, exactly as today:
its rebased prefix LUT, signature map, norm table and counter-stripped suffix
records, each its own allocation. N images are N independent objects on the
same device; nothing is interleaved, so an image is uploaded and freed on its
own and a round can be any subset. A `DeviceView` (pointers + `ProbeParams`)
per database is what the kernels receive, as an array of up to `MAX_DBS` (16)
views passed by value.

**Several devices.** The existing Phase 5 split applies *per database*: device
d holds shard d of **every** resident database, the records whose p-symbol
prefix lies in range d. The ranges are the same for all databases of a round
(one *joint* plan, `KmcRaIndex::jointShardPlan`, balancing the summed
per-prefix record counts of the round's databases; the per-prefix counts come
from each `.kmc_pre` alone, so no suffix array has to be in host RAM to plan).
A common cut is what lets the device-side router stay unchanged: the owner of
a query is a function of its prefix only, independent of which database it is
asked of. Within a device the N shards are N independent images, as above.

**Per slot** (the pipeline's ring of `--streams` slots, section 5), the only
layout change is the hit mask: one `uint32` word per 32 positions **per
database**, stored database-major (`hit[db][word]`, stride = the slot's output
word capacity), so each database's mask is contiguous for the D2H copy and the
consumer. The validity mask is one per slot: validity is a property of the
reference, not of the database.

## 4. Kernels: one K1, N K2

All kernel bodies stay in the host/device-shared headers
(`src/gpu/ProbeCore.hpp`, `src/gpu/EncodeProbeCore.hpp`), so the CPU build
instantiates and tests exactly what the GPU runs.

**Signature once.** The KMC2 probe is signature → bin → two LUT loads → binary
search. The minimizer signature of a k-mer depends on k and the signature
length only (the norm table is a pure function of the signature length), not
on the database; when the round's databases share them (the usual case: KMC
chooses `-p` per database but the signature length is its default 9) the
signature is computed once per query and reused for every database. The probe
is split into `signatureOf` + `probeOneWithSig`; `probeOne` (the single-database
oracle validated by `tests/phase1_test.cpp`) is the composition and is
unchanged.

**One device, fused (`encodeProbeWordMulti`).** The existing fused kernel is
one thread per 32-position output word: a rolling 2-bit encode over the word's
positions, each canonical code probed from registers. The multi-database body
is the same loop with, per emitted position, `for d in 0..n: if probe(d, code)
hits[d] |= bit`. The n accumulators live in a per-thread local array (local
memory, L1-resident; 64 bytes for 16 databases), written out once at the end
to `hit[d][w]`; the validity word is computed once. So the encode (K1) and the
signature scan run once per position, and only the LUT lookups and the binary
searches (K2) run n times. The probe functor holds n `DeviceView`s and
switches per database on the record width (32- or 64-bit suffix records are a
per-database property: `(k − p)/4 ≤ 4` bytes), a warp-uniform branch. The
single-database kernel is kept as is and selected when n = 1, so the measured
single-run path is untouched.

**Several devices, routed.** The Phase 5 path is: the encoder runs K1 over the
batch (`routeKernel`: codes to a flat array + owner counts), an exclusive scan
of the counts, a partition into one segment per owner (`partitionKernel`),
then each owner probes its segment in place over NVLink/P2P
(`probeSegmentKernel`, one flag per query) and the encoder scatters the flags
into the position-indexed mask (`scatterHitsKernel`). With n databases, K1,
the scan and the partition are unchanged (the common cut makes the owner
database-independent); `probeSegmentKernel` writes **n flags per query** (one
flag lane per database, `flags[db][query]`), probing its segment against its n
shards with the signature computed once; `scatterHitsKernel` loops over the n
lanes into the n masks. The queries cross the inter-device link once per
batch, not once per database.

**Not generalised:** `--sort-queries` (the CUB sort between K1 and K2) and the
`--probe=hash` backend remain single-database options; panel mode with either
is refused with a message. Both were ablations that did not pay end-to-end.

## 5. What changes in the pipeline slots

The ring of `--streams` slots (`EncodeProbeEngine`) keeps its shape: pinned
host staging (`h_seq`, `h_inv`, `h_desc`), device batch buffers per device,
one CUDA stream per (device, slot), completion events, the submit/wait
protocol and the pack/consume tasks on the thread pool. Changes:

- **Hit masks:** `h_hit` and `d_hit` become n masks (database-major); the D2H
  copy of the slot returns n + 1 masks. `HostSlot::hitWords(d)` addresses them.
- **Routed path:** the flag buffer on the encoder grows to n lanes (n bytes
  per batch position with peer access; n × G lanes when copies are staged
  through host memory); the staged owner-side flag buffer likewise.
- **Capacity estimate** (`estimateDeviceBytes`) takes n, so the batch
  auto-halving that keeps the buffers beside the resident images stays exact.
- **Consumer:** the per-window task runs the statistics pass n times, once per
  database, over the same validity mask and the database's hit mask, into n
  `Window` records. The pack task is unchanged and runs once per batch.
- **Nothing else** in `submit` changes: one H2D of the packed batch per slot
  (to the encoder, with several devices), the same event chain.

## 6. Memory budgeting and rounds

Per device, the resident set must leave room for the pipeline:

```
free_d  >=  sum_i image_i / G                   (G devices: each holds 1/G of every image)
          + depth x slot(n)                     (pipeline buffers, section 5)
          + margin (256 MB: context, kernels, fragmentation)

slot(n) = seq (B/4) + inv (B/8) + descs + valid (B/8) + n x hit (B/8)          (one device)
        + routed extras with G > 1: codes (8B) + partition (12B) + flags (n.B) [+ staged copies]
        for a batch of B bases (default 128 MB; halved until it fits, never below the largest window)
```

An image's size is known before anything is uploaded: `.kmc_pre` size (LUT +
signature map) + `total_kmers × (4 or 8)` from the header, so rounds are
planned up front from the headers alone. Databases are taken in command-line
order; a round is the longest prefix of the remaining list whose images fit
beside the pipeline buffers (`--resident=N` caps it, which the measurement
script uses to force rounds). A single database whose image does not fit even
with the split is refused as today ("use more devices").

Worked examples, the 2×10⁹-k-mer barley-sized image (15.3 GB), 128 MB batches,
3 streams, one device: slot(1) ≈ 0.07 GB, slot(4) ≈ 0.12 GB, slot(8) ≈ 0.18 GB.
A100-80GB (≈ 79 GB free): 5 images (76.5 GB) leave < 3 GB, so the planner takes
4 (61.2 GB) and runs N = 8 in two rounds. A100-40GB: 2 per round, four rounds.
Two A100-40GB with the split: each device holds half of every image, so 4 to 5
per round.

Rounds re-pack the reference (the packed batches are not kept between rounds:
a persistent packed image of a 2 Gbp reference would be ~0.6 GB of host
memory and is the obvious next step if rounds turn out to dominate, see
section 8), but never re-read or re-window it.

Across references (a reference folder with several FASTA files): when the
previous reference ran as one round over the same databases, the images stay
resident and are reused, as a single run reuses its image across references.
Otherwise every resident image is released before planning, so the plan sees
the device's real free memory and an upload never lands on top of an old
image. One device uploads each image exactly as a single run does
(`initGpuIndex` without a plan); the joint cut is used only with several
devices.

## 7. Host side, byte-identity, CPU path

`KmerPanel` (`src/KmerPanel.hpp`) owns the N `KmerDatabase` objects. Per
reference it reads and windows once, plans the rounds, and for each round
either drives the generalised GPU pipeline or the CPU panel loop:

- **GPU:** the pipeline driver (`runGpuPipeline`) is the existing one with a
  consumer that receives the slot's n masks; database i's `Window` records are
  filled by `BitStats::run` over `(hit[i], valid)` exactly as
  `processReferenceGpu` fills them for one database.
- **CPU:** per window, the rolling codec produces the canonical codes once
  (`KmerCodec::getCanonicalCodes`), then for each database the same
  `CKMCFile::IsKmer` loop with the same `addVariationOriginal` run-length
  accumulation runs over those codes: the single-database `getStatsFromSequence`
  with its probe loop lifted into a per-database loop. The encode amortizes,
  the probes do not. The round size on the CPU is bounded by host RAM (each
  random-access open copies that database's suffix array into RAM:
  `--resident` caps it).

Byte-identity: a TSV row is `(id, start, end, totalKmers, observedKmers,
variations, kmerDistance)`. The first three come from the windowing, done once
and identically; the last four from the per-database hit stream over the same
validity stream. The GPU hit stream for database i is `probeOne` over the same
codes as a single run (shared template; `tests/phase6_test.cpp` checks the
multi-database body word-for-word against n single-database bodies), the CPU
hit stream is the same `IsKmer` call. The writer is the same function. Hence
the files are identical, which the gate (section 9) checks rather than assumes.

Existing command lines are untouched: without `--db`/`--db-list` the tool runs
the single-database code path exactly as before (same functions, same output,
same log lines). `fastibsmapper` and `KDBIntersect` do not take panel options.

## 8. Cost model

Let L = index read+repack+upload per database, R = reference read, P = pack
wall time per reference pass, S = statistics wall time per (database,
reference), D = device time per database per reference pass, W = write per
(database, reference). Single runs: `T1 = L + R + max(P + S, D) + W` per
sample (the pipeline overlaps host and device work; today the host side wins:
P + S ≈ 2.2 s of 16-way work vs D ≈ 1.5 s, the 2×10⁹ positions at the
measured 1.33 Gq/s of the ported-index probe). D per database is kept at its
single-run value below, which is conservative: in panel mode the encode and
the signature scan are shared, so a database's marginal device time is less.

Panel of N with resident capacity C per round, per reference, a round of n
databases costing `max(P + n·S, n·D)`:

```
T_N        = R + N·L + N·W + sum over rounds of max(P + n·S, n·D)
per sample = T_N / N  ~=  L + W + R/N + max(P/n + S, D)
```

| phase | per sample, single runs | per sample, panel | amortizes? |
|---|---|---|---|
| index read+repack+upload (L) | 4.2 s | 4.2 s | **no**: each database must be loaded once |
| FASTA read (R) | 4.1 s | 4.1 / N | **yes**, over the whole panel |
| packing (P, 20 thread-s) | 1.25 s (host) | 1.25 / n | **yes**, over a round |
| statistics (S, 15 thread-s) | 0.94 s (host) | 0.94 s | **no**: per (database, window) |
| device probing (D) | hidden under P + S | ≈ 1.5 s per database, in the open once n·D > P + n·S (n ≥ 3) | no |
| write (W) | 0.09 s | 0.09 s | no |

Predicted on the A100 numbers above: N = 8 in two rounds of 4, per sample
≈ 4.2 + 0.09 + 0.51 + max(0.31 + 0.94, 1.5) = **6.3 s vs 10.5 s (1.7×)**;
N = 2 in one round ≈ 4.2 + 0.09 + 2.05 + max(0.63 + 0.94, 1.5) = 7.9 s (1.3×);
N = 4 in one round ≈ 4.2 + 0.09 + 1.03 + 1.5 = 6.8 s (1.5×). The floor as N
grows is `L + W + D ≈ 5.8 s`: once the reference side is amortized, a panel
is paced by the per-database index preparation (4.2 s) and the device's N-fold
probing (1.5 s per database). Two consequences the design makes explicit:

1. **The next lever is the index load.** The upload (0.8 s) can overlap the
   previous round's scan on its own stream; the read+repack (CPU work on the
   pool's cores) competes with packing and statistics, so it gains only on a
   node with spare cores or I/O-bound reads. Prefetching the next round's
   images is listed as future work, not implemented here.
2. **The device becomes the pace-setter at n ≥ 3 resident databases**, which
   is where the Phase 3 probe-structure ablations (hash: 4.9× probe
   throughput, at 1.7× the footprint and a 4.9 s build) would start to pay,
   the opposite of the single-run conclusion.

The CPU path does not share these proportions: there the probe loop is 92 %
of the scan (CPU-OPT 128.6 s = RA open 7.1 + read 3.7 + probe/statistics
117.6) and is per database, so panel mode on the CPU amortizes only the read
and the codec, a few per cent. Panel mode is a GPU design; the CPU path
exists so the same command runs everywhere and so the WSL build gates it.

## 9. Validation and measurement

- `tests/phase6_test.cpp` (CPU build): for n ∈ {2, 3, 5} databases of the
  panel corpus, the multi-database fused body's n hit words and validity word
  equal n runs of the single-database body, on every output word of every
  window shape the Phase 2 test uses; `probeOneWithSig` composes to
  `probeOne`; the joint shard plan partitions the prefix space and each
  database's shards hold its records exactly once.
- `scripts/make_test_panel.sh <out> <N> [k] [ref_bp] [seed]`: one reference,
  N databases built from N differently-seeded mutated copies of it (SNPs and
  dropout blocks differ per database), so the N hit patterns differ.
- `scripts/gate_multidb.sh <fastibs> <corpus> [options]`: for N ∈ {2, 3, 5},
  W ∈ {10000, 50000}: N single runs vs one panel run, `cmp` on every TSV;
  plus a forced-rounds pass (`--resident=2`, through `--db-list`) and two
  references in one folder (resident reuse and release). Run for k = 25 and
  k = 31. The CPU-only build and the CUDA build without a device both take
  the CPU panel path; on a GPU node the same script gates the device path,
  which `scripts/slurm_multidb_gate.sbatch` runs with `--gpus=1`, `--gpus=2`
  (the routed kernels and the joint cut) and `--cpu`.

## 10. Status and the local check of the cost model

Implemented and gated on the CPU (WSL, 2026-10-28): the CPU-only build and
the CUDA build without a device pass `tests/phase6_test.cpp` and
`scripts/gate_multidb.sh` at k = 25 and k = 31, and single runs stay
byte-identical to the previous build's. On the GPU (Ibex job 54696020,
2×A100-SXM4-80GB, 2026-10-28) `scripts/slurm_multidb_gate.sbatch` passed: the
same gate with `--gpus=1` (fused multi-database kernel), `--gpus=2` (routed
kernels, joint cut) and `--cpu`, at k = 25 and k = 31, 48 byte-identical TSVs
per configuration.

The laptop has no GPU, so the local run checks the model's structure on the
CPU path, not its GPU numbers. Setup: a 10 Mbp reference and four 10.2×10⁶-k-mer
databases (`make_test_panel.sh panel_big 4 25 10000000 11`), W = 50 000, 4
threads, the CPU path (`--cpu`), medians of 9.

| N | single run, s per sample | panel, s per sample | read per database, panel | statistics per database, panel |
|---|---|---|---|---|
| 1 | 2.13 | 2.19 | 0.020 s | 2.02 s |
| 2 | 2.19 | 2.08 | 0.010 s | 1.97 s |
| 4 | 1.98 | 1.98 | 0.0045 s | 1.87 s |
| 8 | 2.07 | 1.95 | 0.0021 s | 1.86 s |

What the model predicts for this CPU run and what it showed:

- **Amortized: the reference read**, 0.019 s per database in single runs,
  falls as 1/N in the panel (0.020, 0.010, 0.0045, 0.0021 s), as predicted.
- **Not amortized: the statistics**, the IsKmer loop per database, stay at
  1.9 to 2.0 s per database at every N, as predicted. The codec, the only
  per-window work the CPU panel shares, is 1.1 % of it (0.09 s vs 8.2 s
  single-threaded on the 10.2 M positions).
- **Per sample**, the model predicts a saving of (R + C/T)(1 − 1/N), at most
  0.04 s or 2 %. The measured medians move by −6 % to +3 %, inside the
  run-to-run spread (1.8 to 3.1 s per sample at N = 1): no measurable gain,
  which is the prediction. The index setup grows mildly with N in the panel
  (0.031 s per database single, 0.049 s at N = 8) because N random-access
  views are held at once.
- All 135 panel outputs of the run (9 repetitions × 15 databases) were
  byte-identical to the single runs'.

The GPU prediction (section 8: 10.5 → 7.9, 6.8 and 6.3 s per sample at
N = 2, 4 and 8 on one A100) rests on the measured A100 breakdown, in which the
amortizable part (FASTA read 4.1 s, packing 1.25 s) is half of a scan instead
of 1 % of it.

**Measured on one A100-SXM4-80GB (Ibex job 54696030, 2026-10-28)**: three
barley databases (1.65–2.06×10⁹ k-mers, k = 31) repeated to eight, MorexV3,
W = 50 000, medians of 2, every panel TSV byte-identical to the single runs'.

| N | single, s per sample | panel, s per sample | predicted | speed-up |
|---|---|---|---|---|
| 1 | 12.07 | 11.97 | 10.5 | 1.0× |
| 2 | 11.66 | 7.85 | 7.9 | 1.49× |
| 4 | 12.13 | 6.12 | 6.8 | 1.98× |
| 8 | 11.32 | 5.36 | 6.3 | 2.11× |

The model was conservative in two places. Setup is not purely per database:
a panel database costs 2.9 s of setup against 3.5–3.9 s for a single run,
because process and device start-up are paid once. And the 8.9 GB barley
images fit eight to an 80 GB device, so N = 8 ran in one round, not two. The
scan phase per database falls to 1.7 s at N = 8, close to the device term
D = 1.5 s, so the device does set the pace once four or more databases are
resident. The measured floor (setup + scan + write) is about 4.7 s per
sample; the index setup remains the next lever.
