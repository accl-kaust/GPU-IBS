# From clone to a two-GPU run on 100 GB of real reads

This is the end-to-end recipe for GPU-IBS on a real sequencing run: build the tools, download a ~100 GB public barley read set and its reference, build the KMC database, check which GPU it fits, and run `fastibs` on two GPUs. Every step is a script under [`scripts/realdata/`](../scripts/realdata/); this page says what each one does, what it needs, and how to read the result.

## Command sheet

```bash
git clone https://github.com/accl-kaust/GPU-IBS.git GPU-IBS && cd GPU-IBS
bash scripts/realdata/00_setup.sh                                   # conda env, GPU-IBS build (+ the original FastIBS in ../FastIBS-orig; SKIP_ORIG=1 skips it)
DATA=/path/with/500GB/free/FastIBSData
tmux new -d "bash scripts/realdata/01_get_data.sh $DATA ERR6098905 2>&1 | tee getdata.log"   # 100 GB reads + reference (hours)
sbatch [-p <cpu partition>] scripts/realdata/02_build_kmc.sbatch $DATA ERR6098905 25 2   # KMC database (CPU job, 1-2 h) + footprint pre-flight
bash   scripts/realdata/03_preflight.sh $DATA/kmc_sets/ERR6098905   # which GPU it fits (re-run any time)
sbatch [-p <gpu partition>] --gres=gpu:a100:2 scripts/realdata/04_run_2gpu.sbatch  $DATA ERR6098905 50000   # the two-GPU run
```

## 0. Assumptions and the one hard constraint

- **Cluster.** SLURM, nodes with two NVIDIA GPUs (the scripts' headers ask only for `--gres=gpu:2`, CPUs, memory and time; add your partition, account and GPU type on the `sbatch` line, e.g. `sbatch -p <partition> --gres=gpu:a100:2 ...`), conda (Miniforge) available, a login or data-transfer node with internet access, and about 500 GB of scratch (100 GB reads, 200–300 GB of KMC temporary bins, ~10 GB database, 4.3 GB reference).
- **"100 GB dataset."** One ENA run of 99.67 GB gzipped FASTQ = 213 gigabases of reads (Section 3). If you meant 100 gigabases, the same run is about twice that; nothing in the recipe changes.
- **The index must fit the GPUs used, combined.** With one device GPU-IBS keeps the whole KMC index resident; with `--gpus=2` (the default when two are visible) the index is split across the devices by k-mer prefix range, each holding about half (see Phase 5 in [`docs/PHASES.md`](PHASES.md)), so a database has to fit their combined memory. Footprint: `stride × N + 8 × 512 × 4^p`, with N the number of k-mers, p the prefix length KMC chose, and stride the (k−p)/4 suffix bytes widened to 4 or 8. Measured for ERR6098905 at k=25 with `-ci2`: N = 2,055,207,296 and p = 9, so the stride is 4 bytes and the device image is 9.30 GB; it fits one 16 GB V100. (Before the database existed this page projected 4–5 × 10⁹ k-mers and 32–40 GB; a highly repetitive genome at 50× leaves far fewer distinct k-mers once singletons are dropped.) `03_preflight.sh` computes the exact footprint from the database and gives the verdict before any GPU job starts. Wheat (15 Gbp, ~10¹⁰ k-mers) is why this recipe uses barley.
- **Multi-GPU semantics.** One process with `--gpus=2` splits the index across the two devices: it buys capacity, and on a single reference it is about as fast as one device because the scan is host-bound (sample 40 on the V100 node: 31.1 s on one GPU, 30.2 s on two shards). The run that shortens the wall time is two processes, one per device, each on half of the chromosomes (25.4 s); `fastibs` writes one TSV per reference FASTA file, so the two halves together are exactly the output of a whole-reference run. Steps 3 and 6 are built around that.

## 1. Clone

```bash
git clone https://github.com/accl-kaust/GPU-IBS.git GPU-IBS
cd GPU-IBS
```

All scripts anchor themselves at the checkout (`SLURM_SUBMIT_DIR`, or `GPUIBS_DIR=` to override), so submit jobs from this directory.

## 2. Build GPU-IBS and the original FastIBS: `00_setup.sh`

```bash
bash scripts/realdata/00_setup.sh          # ~15 min the first time (conda solve + two builds)
```

What it does:

1. Creates the conda env `gpuibs` if it is missing (cmake, ninja, boost, zlib, kmc, aria2, gcc/g++ 12, CUDA toolkit 12.4). An existing env is left alone.
2. Builds GPU-IBS natively: `FASTIBS_CUDA=on bash compile.sh` → `bin/{fastibs,fastibsmapper,KDBIntersect,probe_bench}` for sm_70/80/90.
3. Clones the upstream repository (github.com/githubcbrc/FastIBS) into `../FastIBS-orig` and builds it with the same toolchain, so you can compare GPU-IBS with the original tool on your data (`SKIP_ORIG=1` skips this). Upstream's CMake hard-codes `/project/KMC/bin/libkmc_core.a` (its Docker mount) and `cmake_minimum_required(3.0)`, which CMake 4 refuses; the script patches those two lines and adds `-DBOOST_TIMER_ENABLE_DEPRECATED` for `boost/progress.hpp`. The sources are untouched, so `../FastIBS-orig/bin/fastibs` is the original algorithm and the original CLI (exactly four positional arguments).

Container alternative: `bash install.sh` builds `fastibs.sif` (via Docker, or directly with Apptainer when Docker is absent) and `bash run.sh <RUN> $DATA 50000 --gpus=2` runs the same binary inside it with `--nv`. The scripts below use the native `bin/` binaries.

## 3. The dataset and how to get it: `01_get_data.sh`

**Reads: ENA run [ERR6098905](https://www.ebi.ac.uk/ena/browser/view/ERR6098905)**, *Hordeum vulgare* (barley), genotype "40" of the IPK Gatersleben SHAPE panel, study [PRJEB45512](https://www.ebi.ac.uk/ena/browser/view/PRJEB45512) "Raw sequence data of 47 diverse barley (Hordeum vulgare L.) genotypes" (~50× Illumina, public since 2023-11-17).

| | |
|---|---|
| Instrument, layout | Illumina NovaSeq 6000, paired 2×250 bp |
| Reads, bases | 1,416,842,440 reads, 212,985,590,007 bases (~50× of the 4.2 Gbp genome) |
| Files | `ERR6098905_1.fastq.gz` 48,867,491,666 B, `ERR6098905_2.fastq.gz` 50,807,061,323 B = **99.67 GB** |
| MD5 | `69d9e957635095b300b8221c8326caed`, `de8eb14ddd92fc83a383e3270dfffac9` |
| Source | `https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR609/005/ERR6098905/` |

The same sample has a second run of 97 GB (ERR6098906); pass it as the second argument in a second invocation and append its files to `reads/ERR6098905.files` if you want a ~100× database. Any other ENA run accession works too: sizes, MD5s and URLs are read from the ENA file report at run time and verified after download.

**Reference: MorexV3 pseudomolecules** (barley cv. Morex), chromosomes 1H–7H from Ensembl Plants (`ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/hordeum_vulgare/dna/`, per-chromosome `dna.primary_assembly.<chr>.fa.gz`, 150–194 MB each, verified against the directory's `CHECKSUMS`). Decompressed: about 4.3 GB, one file per chromosome named `MorexV3_<chr>_genome.fasta`. Two details of `fastibs` decide those names: it only picks up files whose extension is exactly `.fasta` (a `.fa` or `.fa.gz` is silently skipped), and it strips a trailing `_genome` when naming the output, so results are called `ERR6098905_v_MorexV3_1H_50000.tsv`.

```bash
DATA=/scratch/$USER/FastIBSData          # anything with ~500 GB free
tmux new -d "bash scripts/realdata/01_get_data.sh $DATA ERR6098905 2>&1 | tee getdata.log"
```

Resumable (aria2c with 8 connections, or `wget -c`), idempotent, verifies size and MD5, and finishes by splitting the reference into `reference_gpu0/` and `reference_gpu1/` (relative symlinks; the exact two-way partition that minimizes the larger half, which for the seven barley chromosomes is 2H+3H+7H against 1H+4H+5H+6H: the larger half is 54 % of the bases, so the ideal two-process gain on the processing part is 1.84×). `STEP=reads|ref|split` runs one part. Expect one to several hours for the reads depending on your link to EBI.

If the ENA FTP is slow (it can throttle to KB/s), the same runs are on the NCBI SRA public S3 mirror as one `.sra` file per run: convert it with `fasterq-dump` from sra-tools (about 20 minutes for ERR6098905 on fast local disk) and count the FASTQ files with the KMC flags of Section 4. Where the `ftp.ensemblgenomes.ebi.ac.uk` host is refused, point `ENSEMBL_DIR=` at another mirror of the same directory.

Resulting layout (what `run.sh` and the original FastIBS expect, plus the two halves):

```
$DATA/
├── reads/            ERR6098905_1.fastq.gz  ERR6098905_2.fastq.gz  ERR6098905.files
├── kmc_sets/         ERR6098905/ERR6098905.res.kmc_pre|.kmc_suf     (step 4)
├── reference/        MorexV3_1H_genome.fasta ... MorexV3_7H_genome.fasta
├── reference_gpu0/   -> ../reference/... (about half the bases)
├── reference_gpu1/   -> ../reference/... (the other half)
└── FastIBS_runs/     results
```

## 4. Build the KMC database: `02_build_kmc.sbatch`

```bash
sbatch [-p <cpu partition>] scripts/realdata/02_build_kmc.sbatch $DATA ERR6098905 25 2      # k=25, count >= 2
```

A CPU job (`--cpus-per-task=32 --mem=200G`, 24 h limit; the header names no partition) that runs `kmc` from the conda env (there are no KMC binaries in this repository):

```
kmc -k25 -ci2 -cs255 -m<75 % of --mem> -t32 -fq -hp -j kmc_stats.json @reads/ERR6098905.files kmc_sets/ERR6098905/ERR6098905.res <scratch>
```

then `kmc_tools info` and the pre-flight of Section 5. k=25 matches the recorded corpora; `-ci2` drops k-mers seen once (sequencing errors), `-cs255` is the one-byte counter the index format assumes. Temporary bins need roughly 2–3× the compressed input: set `KMC_TMP=` to local NVMe or a scratch filesystem, not your home. Expect one to two hours on 32 cores with fast scratch (recorded: 1,036 s with 64 threads on node-local NVMe, job 68887; 2,055,207,296 k-mers, 10.3 GB on disk). About k=31 (IBSpy's default): run the pre-flight on the result; the 4-byte stride of the recorded k=25 databases comes from KMC choosing p=9, and a larger k needs more suffix bytes per k-mer.

## 5. Pre-flight: `03_preflight.sh`

```bash
bash scripts/realdata/03_preflight.sh $DATA/kmc_sets/ERR6098905            # on a GPU node: verdict for this device
bash scripts/realdata/03_preflight.sh $DATA/kmc_sets/ERR6098905 81920      # elsewhere: verdict for an 80 GB A100
```

Reads k, N and the prefix length from `kmc_tools info`, prints the device footprint with the formula of Section 0, the host RAM the load needs (the loader materializes the widened image in RAM before the upload, plus the page cache of `.kmc_suf`), and FITS / DOES NOT FIT against the device, with a 4 GiB margin for the pipeline's buffers. Step 6 calls it with `REQUIRE_FIT=1` and aborts instead of letting the upload fail halfway. Validation: on the recorded 2,000,199,169-k-mer corpus (k=25, p=5) it prints 15,266 MiB = 16.01 GB, the value the tool reports at upload.

Pick the GPU type from the verdict and ask for it on the `sbatch` line (`--gres=gpu:<type>:2`, plus your partition). The sample-40 database (9.30 GB) fits a 16 GB V100 as well as a 40 GB or 80 GB A100.

## 6. Run on two GPUs: `04_run_2gpu.sbatch`

```bash
sbatch -p <partition> --gres=gpu:a100:2 scripts/realdata/04_run_2gpu.sbatch $DATA ERR6098905 50000            # MODE=split (default)
MODE=both sbatch -p <partition> --gres=gpu:a100:2 scripts/realdata/04_run_2gpu.sbatch $DATA ERR6098905 50000   # also --gpus=2, and diff the two
```

`MODE=split` starts two `fastibs` processes at once, `CUDA_VISIBLE_DEVICES=0` on `reference_gpu0` and `CUDA_VISIBLE_DEVICES=1` on `reference_gpu1`, both writing into `$DATA/FastIBS_runs/ERR6098905_w50000_split/`; the job log shows each process's `K-mer path: ... (GPU-BS)` line, its `index read+repack` and `Reference processed` times, and the wall time from the first start to the last exit. `MODE=replica` is the single process with `--gpus=2`, which splits the index across the two devices; `MODE=both` runs both and byte-compares the two result folders (they are identical). Header: `--gres=gpu:2 --cpus-per-task=16 --mem=200G` (in split mode each process holds its own host image while loading, about 8 GB for sample 40; the pre-flight prints the exact requirement).

The same run through the container: `bash run.sh ERR6098905 $DATA 50000 --gpus=2` for the replica mode, and for the split mode two `apptainer exec --nv --bind $PWD:/project,$DATA:/project/data fastibs.sif /project/bin/fastibs /project/data/kmc_sets/ERR6098905 /project/data/reference_gpu<i> /project/data/FastIBS_runs 50000 --gpu` with `CUDA_VISIBLE_DEVICES=<i>` (the relative symlinks in `reference_gpu*` resolve inside the bind).

## 7. Time and storage budget

| step | wall time (order of magnitude) | resources |
|---|---|---|
| 00 setup | 15 min | login node or build allocation |
| 01 data | 1–5 h (network-bound) | 105 GB of disk, internet |
| 02 KMC | 1–2 h | 32 cores, 200 GB RAM, 200–300 GB scratch; database ~10 GB |
| 04 two-GPU run | 1–2 min | 2 GPUs, 200 GB RAM |

## 8. When something is off

- `DOES NOT FIT` in the pre-flight: use an 80 GB device, or a genome ≤ 3 Gbp (a 30× human or maize run of the same size gives a ~24 GB index that fits a 40 GB A100; the scripts only need another ENA accession and reference URL pattern).
- `fastibs` prints `[gpu] no CUDA device found ... (cudaGetDeviceCount: <reason>)` and takes the CPU path: `no NVIDIA driver is loaded` or `no CUDA-capable device is detected` inside a container means no `--nv`; `unknown error` means a broken device, move nodes.
- `index read+repack` far below ~2 GB/s in a timed pass while the warm-up was fast: the page cache was lost between passes (shared node under memory pressure, or a filesystem that does not cache); re-run on another node, the gates' `tests/load_bench.cpp` (`load_bench <db> 0 0`) is the diagnostic.
- Out-of-memory killing a GPU pass: the host image is `stride × N` bytes per process plus the cached `.kmc_suf`; the pre-flight prints the `--mem` to request, doubled for the split mode (two processes).
- Reference files not picked up: the extension must be exactly `.fasta` (the download script names them that way); gzip-compressed references are not scanned even though the reader could decompress them.
- Errors while processing a reference: the tool writes the details to `log.txt` in the working directory, prints `<n> reference(s) failed; see log.txt` on stderr and exits 1 (`04_run_2gpu.sbatch` also shows the tail of `log.txt`).
