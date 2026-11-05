# GPU-IBS tool reference

Everything the three tools accept and produce: the GPU path and its options, the environment variables, input checks and exit status, and the full `--help` text and output format of `fastibs`, `fastibsmapper` and `KDBIntersect`. The [README](../README.md) has the quick start and the usual workflow, [DOCKER.md](DOCKER.md) the container image.

The `--help` texts below show the tools as `/project/bin/<tool>`, their path in the original FastIBS container flow; each tool prints its own path.

Contents: [GPU execution](#gpu-execution) · [Environment variables](#environment-variables) · [Troubleshooting](#troubleshooting) · [Command-line options](#command-line-options) · [CPU-BATCHED](#cpu-batched) · [Inputs and exit status](#inputs-and-exit-status) · [fastibs](#fastibs-ibs-distance-per-window) · [Panel mode](#panel-mode-many-samples-one-reference-pass) · [fastibsmapper](#fastibsmapper-per-base-k-mer-coverage) · [KDBIntersect](#kdbintersect-shared-k-mers-of-two-databases)

## GPU execution

Nothing changes in how the tools are called. When the binary was built with CUDA and a device is visible, `fastibs` and `fastibsmapper` stream the reference through the GPU and `KDBIntersect` probes the device-resident database; otherwise they run on the CPU. Both paths produce byte-identical output. The device holds the KMC index — 4 bytes per k-mer when the record suffix (k − p bases, p the KMC prefix length) fits 16 bases, else 8, plus its look-up tables: 16 GB for the 2×10⁹-k-mer synthetic `corpus_big`, 8.9 GB for each 2.06×10⁹-k-mer barley database of the evaluation — and, with the hash backend, 8/ℓ bytes per k-mer instead.

**k-mer length.** The GPU paths, the rolling codec (CPU-OPT) and the batched CPU probe hold a k-mer in one 64-bit word, so they need **k ≤ 32** (the paper's runs use k = 25 and 31). A database with k > 32, which KMC allows, runs the original string-based path (CPU-BASE, as under `FASTIBS_LEGACY=1`) automatically in all three tools, with one line saying so (`[fastibs] k=35 > 32: using the original string path (CPU-BASE); ...`); the GPU and `--probe` options are then ignored. Its output is that of the original tool. In panel mode such databases are scanned one at a time on the same path, writing the same files. **With several visible GPUs the index is split across them**, not replicated: device *d* holds one contiguous range of k-mer prefixes (~1/N of the records and of the look-up tables, balanced by record count) and searches only its own k-mers, so a database has to fit the *combined* memory of the devices used, and the per-device footprint shrinks as devices are added. One device holds the whole index.

### Environment variables

All optional:

| Variable | Effect |
|---|---|
| `FASTIBS_GPU=0` | Force the CPU path (also honoured by `run.sh`/`intersect.sh`, which then omit `--nv`) |
| `FASTIBS_GPUS=N` | Use N of the visible devices (default: all of them; `--gpus=N` overrides). The only device-count knob for `KDBIntersect`, which has no options; `CUDA_VISIBLE_DEVICES` still selects *which* devices |
| `FASTIBS_LEGACY=1` | Use the original FastIBS string-based CPU code (kept as the correctness oracle; taken automatically for k > 32) |
| `FASTIBS_CPU_PROBE=batched` | On the CPU path, probe with K k-mers in flight per thread over the host image of the index (CPU-BATCHED, the same as `--probe=batched`; [`docs/CPU_BATCHED_PROBE.md`](CPU_BATCHED_PROBE.md)) instead of one at a time through the KMC API (CPU-OPT, the default) |
| `FASTIBS_CPU_BATCH=K` | Probes in flight per thread for CPU-BATCHED (default 16, at most 256) |
| `FASTIBS_GPU_SYNC=1` | Per-window synchronous GPU probing instead of the streamed pipeline |
| `FASTIBS_LOAD_THREADS=N` | Threads for reading the on-disk index into the device image (default: the CPUs in the process's affinity mask, i.e. the SLURM allocation) |
| `FASTIBS_THREADS=N` | Size of the CPU thread pool for window tasks, packing and statistics (default: the node's hardware concurrency, as in the original), and the number of threads `KDBIntersect` enumerates the smaller database on (default there: the CPUs in the process's affinity mask, i.e. the SLURM allocation; clamped to 256; `1` is the sequential walk) |
| `FASTIBS_INTERSECT_BATCH_M=N` | `KDBIntersect` only: millions of k-mers per probe batch and per device (default: 32M in a single-threaded run, shared out among the enumeration threads down to 2M) |
| `FASTIBS_PROBE=hash` | `KDBIntersect` only: probe a device hash set of the larger database instead of its ported index (the options' `--probe=hash` for the other tools) |
| `FASTIBS_HASH_LF=F` | `KDBIntersect` only: load factor of that hash set (default 0.6, 0.1 to 0.95) |
| `FASTIBS_NO_PEER=1` | With several GPUs, route the queries through host-staged copies even where the devices have peer access (the same as `--no-peer`) |

### Troubleshooting

The reason printed with the fallback message (`[gpu] no CUDA device found, using CPU path (cudaGetDeviceCount: <reason>)`) tells the common cases apart. `CUDA driver version is insufficient for CUDA runtime version; no NVIDIA driver is loaded: ...` means no driver library could be loaded: a machine without an NVIDIA GPU, or a container started without `--gpus all` (Docker) or `--nv` (Apptainer, Singularity). The same CUDA message followed by `the driver supports CUDA 11.x, ...` means the host driver is older than 525.60. `no CUDA-capable device is detected` means the driver is there but sees no device (no GPU in the allocation, or `CUDA_VISIBLE_DEVICES` hides them); `unknown error` means the device itself is unhealthy, which SLURM confirms with its own `slurmstepd: error: Failed to get device handle for GPU 0` lines on that node, so the job should move elsewhere (`sbatch -x <node>` or `-w <node>`).

### Command-line options

Command-line options (`fastibs` and `fastibsmapper` only; everything after the original positional arguments is optional):

```
Options (GPU builds):
  --gpu            Force the GPU path (default: auto when a device is present)
  --cpu            Force the CPU path
  --batch-mb=N     Sequence megabytes per GPU batch (default 128; halved
                   automatically until the pipeline buffers fit beside
                   the index on the device)
  --streams=N      GPU pipeline depth in streams (default 3)
  --sync           Per-window synchronous GPU probing (Phase 1 path)
  --sort-queries   Sort each batch by k-mer value before probing (GPU-*-SORTED; one device)
  --probe=bs|hash  Probe backend: ported KMC index (default) or hash set
  --probe=batched  CPU option, in every build (implies --cpu): K probes in
                   flight per thread over the host index image
                   (CPU-BATCHED); K = FASTIBS_CPU_BATCH (default 16)
  --hash-lf=F      Hash-set load factor (default 0.6)
  --pin-lut        Keep the prefix LUT in persisting L2 cache
  --gpus=N|all     Devices to use (default: all visible, or FASTIBS_GPUS=N).
                   The index is split across them: each device holds a
                   k-mer prefix range (~1/N of the index); a batch goes to
                   one device, which encodes it and routes the queries to
                   their devices over NVLink / PCIe peer-to-peer; output is
                   identical for any N
  --no-peer        Route the queries through host-staged copies even where
                   the devices have peer access (as FASTIBS_NO_PEER=1)
  -h, --help       Print this text and exit
  --version        Print the GPU-IBS version and build, and exit
```

Numeric options take a whole integer (`--batch-mb=12x` is an error, not 12); `--gpus=0` is rejected, since it could mean every device or none (`--gpus=all` and `--cpu` say which). An unknown option or a bad value prints the usage and exits with status 1.

The defaults (`--probe=bs`, streamed, every visible GPU) are the production configuration; the other options are the ablations of the paper (`--sort-queries` is a single-device option). How the split works: each batch of packed sequence (0.375 bytes per position) goes to **one** device, chosen round-robin so every device encodes 1/N of the batches. That device encodes and canonicalizes all positions, works out each k-mer's owner from its prefix, and partitions the queries into one segment per device; the other devices probe their segments straight out of its memory and write one hit flag per query back — over NVLink where the GPUs have it, PCIe peer-to-peer otherwise (host-staged copies if a pair has no peer access, if enabling it fails, or when forced with `--no-peer` / `FASTIBS_NO_PEER=1`; the startup log prints the topology and one `[gpu] routing: peer` or `[gpu] routing: staged` line, `nvidia-smi topo -m` shows which links are NVLink). Every timed multi-GPU run used peer access over NVLink; `--no-peer` exists so the staged routing can be checked against `--gpus=1` on such a node. The encoder assembles the hit mask and returns it, so the host side is exactly the single-GPU pipeline and the output is identical for any N by construction — every k-mer is held by exactly one device. Per device the work is encode(1/N of the batches) + probe(1/N of the queries) and the upload volume is 1/N of the index; the routing buffers cost about 20+N bytes per position of batch per stream slot on each device (≈1.6 GB per slot at 64 MB batches and N = 4; the pipeline log prints the total). `KDBIntersect` routes the enumerated k-mers to their owning device on the host instead, so each query crosses PCIe once. A single scan of one reference remains largely host-bound (reading the index and the FASTA are about half of its wall time), so the main gains of the split are databases larger than one device's memory and the probe-bound workloads (intersection matrices, `--probe=hash`); independent runs can still be packed one process per GPU with `CUDA_VISIBLE_DEVICES`. The routing buffers of that pipeline take about 21 bytes per batch position per stream on every device; when they do not fit beside the index shard, the batch size is halved automatically until they do (a `[gpu] pipeline buffers:` line reports the requested and the used size). A reference that fails to process is reported on stderr and in `log.txt`, and the tool exits with status 1.

### CPU-BATCHED

**CPU-BATCHED** (`--probe=batched`, or `FASTIBS_CPU_PROBE=batched` on the CPU path) is the CPU-side counterpart of the GPU's probe: each worker thread keeps K membership probes in flight (software pipelining with prefetch over the host image of the index — the same arrays the GPU path uploads) instead of probing one k-mer at a time through the KMC API, so a core overlaps K DRAM misses instead of serializing them. It is the CPU baseline tuned the way the GPU path is; its output is byte-identical to CPU-OPT's, `FASTIBS_CPU_BATCH=K` sets K (default 16), and the design, the local checks and what to measure on a cluster are in [`docs/CPU_BATCHED_PROBE.md`](CPU_BATCHED_PROBE.md).

## Inputs and exit status

The three tools check their inputs before doing any work. A missing database folder, or one without a `<prefix>.kmc_pre` / `<prefix>.kmc_suf` pair, ends in one `Error: ...` line naming the path and exit status 1 (a KMC prefix given in place of its folder gets a hint). So does a missing reference folder or a results path that cannot be created. `fastibs` requires `<windowSize>` to be a whole number larger than the database's k: consecutive windows overlap by k bases, and in the original tool W = k never advanced (memory ran out), W < k gave one bogus window per contig and W = 0 divided by zero. References are the `*.fasta`, `*.fa` and `*.fna` files of `<referencePath>`, each optionally gzip-compressed (`*.fasta.gz`, ...). Every other entry is named as skipped, and a folder with no reference at all is an error. Output names for `*.fasta` files are unchanged: `fastibs` writes `<database folder>_v_<name>_<W>.tsv`, where `<name>` is the file name cut at `_genome` (else the whole file name), and `fastibsmapper` writes `<database folder>_v_<stem>.txt`, where the stem drops `.gz` and the extension. The results folder is created with its parents when missing. Each output is written as `<file>.partial` and renamed once complete. A file that cannot be written (read-only or full volume) fails its reference (stderr and `log.txt`, exit status 1) and leaves no truncated file for the skip-if-exists rule to keep. `-h` / `--help` anywhere on the command line prints the usage on stdout with exit status 0. Exit status 0 means every reference was processed or already had its output; `KDBIntersect` exits 0 with the count on stdout and 1 with an error line on stderr.

## fastibs: IBS distance per window

Running the **fastibs** binary without arguments (exit status 1) or with `-h` / `--help` (exit status 0):
```bash
fastibs --help
```

prints the following.
```bash
FastIBS - IBS Distance Calculator
----------------------------------
Usage:
  /project/bin/fastibs <sourcePath> <referencePath> <resultsFolder> <windowSize> [options]

Arguments:
  <sourcePath>     Path to folder with KMC dataset
                   e.g., /mnt/data/kmc_sets/BW_01002

  <referencePath>  Path to folder with reference genomes (FASTA)
                   e.g., /mnt/data/reference

  <resultsFolder>  Path to folder for storing output results
                   e.g., /mnt/data/FastIBS_runs

  <windowSize>     Length of the sequence window for IBS calculation
                   (integer larger than k, e.g., 50000)

Options (GPU builds):
  --gpu            Force the GPU path (default: auto when a device is present)
  --cpu            Force the CPU path
  --batch-mb=N     Sequence megabytes per GPU batch (default 128; halved
                   automatically until the pipeline buffers fit beside
                   the index on the device)
  --streams=N      GPU pipeline depth in streams (default 3)
  --sync           Per-window synchronous GPU probing (Phase 1 path)
  --sort-queries   Sort each batch by k-mer value before probing (GPU-*-SORTED; one device)
  --probe=bs|hash  Probe backend: ported KMC index (default) or hash set
  --probe=batched  CPU option, in every build (implies --cpu): K probes in
                   flight per thread over the host index image
                   (CPU-BATCHED); K = FASTIBS_CPU_BATCH (default 16)
  --hash-lf=F      Hash-set load factor (default 0.6)
  --pin-lut        Keep the prefix LUT in persisting L2 cache
  --gpus=N|all     Devices to use (default: all visible, or FASTIBS_GPUS=N).
                   The index is split across them: each device holds a
                   k-mer prefix range (~1/N of the index); a batch goes to
                   one device, which encodes it and routes the queries to
                   their devices over NVLink / PCIe peer-to-peer; output is
                   identical for any N
  --no-peer        Route the queries through host-staged copies even where
                   the devices have peer access (as FASTIBS_NO_PEER=1)
  -h, --help       Print this text and exit
  --version        Print the GPU-IBS version and build, and exit

Panel mode (several databases, one reference pass):
  --db=<path>      Another KMC dataset folder to scan in the same pass
                   (repeatable); its output file is named as in a
                   single run. The reference is read and packed once,
                   every batch is encoded once and probed against every
                   resident database; outputs are identical to N runs
  --db-list=<file> A file with one dataset folder per line (# comments)
  --resident=N     At most N databases resident at once (default: as
                   many as fit the device(s); the rest in further rounds)

Notes:
  - All folders should be located on a mounted data volume.
  - Reference files: *.fasta, *.fa or *.fna, optionally gzip-compressed
    (*.fasta.gz, ...); other files in <referencePath> are skipped.
  - k <= 32 takes the accelerated paths; a database with k > 32 runs the
    original string path (GPU and --probe options are then ignored).
  - Exit status: 0 when every reference was processed or already had its
    output; 1 on a bad command line or input, or a failed reference
    (logged to log.txt).

Output:
  A tab-delimited file summarizing IBS distance metrics for each window.
  Columns: seqname, start, end, total_kmers, observed_kmers, variations, kmer_distance
```

Provided a KMC database at `<sourcePath>` , **fastibs** computes IBS distance reports against all references in `<referencePath>`.
The output of **fastibs** is a tab-delimited table with the following columns:

| **Column Name**     | **Description**                                                                 |
|---------------------|---------------------------------------------------------------------------------|
| `seqname`           | The sequence name (typically the accession or identifier of the reference genome segment). |
| `start`             | Start position (0-based) of the genomic window being analyzed.                  |
| `end`               | End position of the window (non-inclusive).                                     |
| `total_kmers`       | Total number of k-mers generated from the reference window. This is usually fixed and depends on the window size. |
| `observed_kmers`    | Number of k-mers found in the sample that match the reference set for that window. |
| `variations`        | Number of variant sites (positions where the reference and sample differ within the window). |
| `kmer_distance`     | Computed IBS distance metric, often reflecting the number of unique k-mers in the reference that are absent from the sample (or vice versa). |

### Panel mode: many samples, one reference pass

A panel scans many sample databases against the same reference(s). Run separately, each sample re-reads and re-packs the reference. With `--db` (repeatable) or `--db-list=<file>`, `fastibs` reads, windows and packs each reference once. It keeps as many databases resident on the GPU(s) as fit, encodes every batch once and probes it against all of them. It writes one TSV per database, with the same name and the same bytes as a separate run:

```bash
fastibs /data/kmc_sets/S1 /data/reference /data/runs 50000 --db=/data/kmc_sets/S2 --db=/data/kmc_sets/S3
fastibs /data/kmc_sets/S1 /data/reference /data/runs 50000 --db-list=panel.txt   # S2, S3, ... one per line
```

Databases that do not fit at once are scanned in further rounds (`--resident=N` caps a round). The reference is still read only once. Without a GPU the same command runs on the CPU, computing each window's k-mers once for all databases. The design and its cost model are in [`docs/MULTIDB_SCAN.md`](MULTIDB_SCAN.md). `scripts/gate_multidb.sh` checks the outputs against separate runs.

## fastibsmapper: per-base k-mer coverage

**fastibsmapper** computes a K-mer mapping of a given KMC base against given references, where each nucleotide position in the reference sequences is associated with a count of how many K-mers (of a fixed size, defined by the kmerSize of the KMC source) overlap that position and exist in the source KMC database.

A quick and easy k-mer mapping can be used to detect regions of high conservation/divergence from a given reference. Even though a KMC database does not retain positional information and only records k-mers and their frequencies from reads, the probability of false positives is small as long as sufficiently long k-mers are used (we use a default of 31 for kmerSize).

Running the **fastibsmapper** binary without arguments (exit status 1) or with `-h` / `--help` (exit status 0):
```bash
fastibsmapper --help
```

prints the following (the Options block is the same as for `fastibs` and is abbreviated here).

```bash
FastIBS - Reference Mapping Tool
---------------------------------
Usage:
  /project/bin/fastibsmapper <sourcePath> <referencePath> <resultsFolder> [options]

Arguments:
  <sourcePath>     Path to folder containing KMC database files
                   (e.g., /mnt/data/kmc_sets/<dataset_name>)

  <referencePath>  Path to folder containing reference genomes in FASTA format
                   (e.g., /mnt/data/reference)

  <resultsFolder>  Destination folder for writing mapping result files
                   (e.g., /mnt/data/FastIBS_runs)

Options (GPU builds):
  ... (same options as fastibs)

Notes:
  - All input folders should reside on a mounted data volume.
  - The tool scans <referencePath> for *.fasta, *.fa and *.fna files, optionally
    gzip-compressed (*.fasta.gz, ...), and processes them against the KMC base;
    other files are skipped.
  - Output filenames follow the format: <sourcePath folder>_v_<reference_stem>.txt
    (the stem without .gz and the FASTA extension)
  - Existing output files will be skipped.
  - Errors during processing are logged to log.txt.
  - k <= 32 takes the accelerated paths; a database with k > 32 runs the
    original string path (GPU and --probe options are then ignored).
  - Exit status: 0 when every reference was processed or already had its
    output; 1 on a bad command line or input, or a failed reference.

Example:
  /project/bin/fastibsmapper /mnt/data/kmc_sets/sample1 /mnt/data/reference /mnt/data/FastIBS_runs

Output: This tool computes a K-mer mapping for the given references, where each nucleotide position in the reference sequences is associated with a count of how many K-mers (of a fixed size, defined by the kmerSize of the KMC source) overlap that position and exist in the source KMC database.
```

The output file holds two lines per reference sequence: the sequence's name (its FASTA header without `>`), then one comma-separated count per base, from 0 (no k-mer of the sample covers the base) to k (every k-mer over it is present). A base inside a run of present k-mers reads k; the counts ramp up and down over the first and last k − 1 bases of a sequence and around a difference:

```
chr1
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,25,25,...
```

The file takes about three bytes per reference base (mostly two digits and a comma), so a 4 Gbp genome gives about 12 GB.

## KDBIntersect: shared k-mers of two databases

This is a simple wrapper around the same KMC API functionality; on a GPU the larger database (by k-mer count, read from the two headers) is probed on the device — split across the visible devices when there are several — with exactly the same count. The smaller database is enumerated straight from its on-disk index on `FASTIBS_THREADS` threads, each walking its own ranges of look-up-table buckets and filling its own probe batches, so the run is probe-bound rather than host-bound; the CPU path (`FASTIBS_GPU=0`, or a CPU-only build) uses the same enumeration with the host probe, and `FASTIBS_LEGACY=1` runs the original single-threaded KMC-API stream-and-probe, which is the oracle (design and gate: [`docs/INTERSECT_PARALLEL_ENUM.md`](INTERSECT_PARALLEL_ENUM.md)).

```bash
K-mer Database Intersection Tool
----------------------------------
Usage:
  /project/bin/KDBIntersect <kDB1Path> <kDB2Path>

Arguments:
  <kDB1Path>     Path to the first KMC database directory
                 (e.g., /mnt/data/kmc_sets/db1)

  <kDB2Path>     Path to the second KMC database directory
                 (e.g., /mnt/data/kmc_sets/db2)

Description:
  This tool computes the intersection size between two KMC databases, i.e.,
  how many k-mers from the first database are present in the second.

Notes:
  - The databases must have the same k-mer size.
  - The tool compares the two KMC databases by k-mer count: the smaller one
    is enumerated and the larger one probed by random access (on a tie, the
    original rule: the .kmc_pre size, then the argument order).
  - The intersection size is output as an integer, which represents the
    number of common k-mers between the two databases.
  - k <= 32 takes the enumeration paths below; databases with k > 32 run the
    original KMC-API stream-and-probe.
  - GPU builds probe the larger database on the visible GPUs, split across
    them when there are several (FASTIBS_GPUS=N limits the count,
    FASTIBS_GPU=0 forces the CPU); the count is identical either way.
  - FASTIBS_PROBE=hash probes a device hash set of the larger database
    instead of its ported index (FASTIBS_HASH_LF: load factor, default 0.6);
    the count is identical, the table costs 8/lf bytes per k-mer.
  - The smaller database is enumerated on FASTIBS_THREADS threads (default:
    the CPUs in the process's affinity mask, at most 256; 1 is the sequential
    walk), on the GPU and on the CPU path alike; FASTIBS_LEGACY=1 runs the
    original KMC-API stream-and-probe instead. The count is identical in
    every case.
  - Exit status: 0 with the count on stdout; 1 on a bad command line or
    database (one error line on stderr).
  - --version prints the GPU-IBS version and build.

Example:
  /project/bin/KDBIntersect /mnt/data/kmc_sets/db1 /mnt/data/kmc_sets/db2
```

The count is the only thing written to standard output (diagnostics go to standard error), so `$(KDBIntersect a b)` captures it directly.
