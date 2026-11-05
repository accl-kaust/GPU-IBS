# GPU-IBS

[![Docker Hub](https://img.shields.io/docker/pulls/ahmadshakleya/gpu-ibs?logo=docker&label=Docker%20Hub)](https://hub.docker.com/r/ahmadshakleya/gpu-ibs)
[![image size](https://img.shields.io/docker/image-size/ahmadshakleya/gpu-ibs/latest?logo=docker&label=image%20size)](https://hub.docker.com/r/ahmadshakleya/gpu-ibs)
[![container build](https://github.com/accl-kaust/GPU-IBS/actions/workflows/docker.yml/badge.svg)](https://github.com/accl-kaust/GPU-IBS/actions/workflows/docker.yml)

GPU-IBS compares sequenced samples with a reference genome and tells you, window by window, how close each sample is to the reference. It is a GPU version of [FastIBS](https://github.com/githubcbrc/FastIBS): the same tools, the same commands and the same output files. If your machine has an NVIDIA GPU, the tools use it; if not, they run on the CPU and give exactly the same results.

The comparison is the k-mer identity-by-state (IBS) scan of [IBSpy](https://github.com/Uauy-Lab/IBSpy). You do not align reads or call variants. Each sample is a [KMC](https://github.com/refresh-bio/KMC) database of the k-mers in its reads, and for every window of the reference GPU-IBS counts how many of the reference's k-mers the sample has and estimates how many bases differ. Windows with a distance near zero are regions where the sample is identical by state to the reference, which is how shared haplotypes and introgressions are traced across a diversity panel.

There are three tools:

- `fastibs` runs the IBS scan and writes one table row per window.
- `fastibsmapper` reports, for every base of the reference, how many of the sample's k-mers cover it.
- `KDBIntersect` counts the k-mers two samples have in common.

On one NVIDIA A100, scanning a sample of 2 billion k-mers against a 2 Gbp reference takes 9.4 seconds, compared with 121 seconds for our optimized CPU code on 16 cores. A real barley sample (2.1 billion k-mers) against the 4.2 Gbp MorexV3 genome takes 21 seconds on one A100, 175 seconds with our optimized CPU code and 311 seconds with the original FastIBS code, both on 16 cores.


## Install

The easiest way is the container image. It contains the three tools and KMC 3.2.4 for counting k-mers.

With Docker (the image is on [Docker Hub](https://hub.docker.com/r/ahmadshakleya/gpu-ibs)):

```bash
docker pull ahmadshakleya/gpu-ibs:latest
docker run --rm --gpus all ahmadshakleya/gpu-ibs:latest gpu-ibs selftest
```

On a cluster, with Apptainer or Singularity (no root needed; replace `apptainer` with `singularity` if that is what your cluster has):

```bash
apptainer pull gpu-ibs.sif docker://ahmadshakleya/gpu-ibs:latest
apptainer exec --nv gpu-ibs.sif gpu-ibs selftest
```

The self-test takes less than a minute. It runs the three tools on a small made-up data set, checks the results against known answers and ends with `SELFTEST PASSED`. It also tells you whether it used the GPU.

A few things to know:

- To use a GPU you need an NVIDIA driver 525.60 or newer, and for Docker also the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On a machine without a GPU, leave out `--gpus all` (Docker) or `--nv` (Apptainer).
- `latest` is the newest version. For an analysis you want to reproduce later, use a fixed version instead, for example `ahmadshakleya/gpu-ibs:1.0.0`.
- You can also build the image yourself: `git clone https://github.com/accl-kaust/GPU-IBS.git`, then `docker build -t gpu-ibs .` inside the folder. Or skip containers and [build from source](#building-from-source).


## Try the example

Run this in an empty folder:

```bash
docker run --rm --gpus all -u $(id -u):$(id -g) -v "$PWD":/data ahmadshakleya/gpu-ibs:latest gpu-ibs example
```

With Apptainer or Singularity, run `apptainer exec --nv gpu-ibs.sif gpu-ibs example` instead. It creates a small project in `gpu-ibs-example/` (a reference and three samples) and runs each tool on it, printing every command before it runs. It is the quickest way to see what the inputs and the results look like.


## Run it on your data

### 1. Set up a project folder

```
my_project/
├── reference/       the reference genome(s): .fasta, .fa or .fna, gzipped or not
├── reads/           the sequencing reads of each sample (FASTQ)
├── kmc_sets/        one folder per sample with its k-mer database (step 3)
└── FastIBS_runs/    the results (created for you)
```

The tools process every FASTA file in `reference/` and write one result file for each, so you can put one genome there or several.

### 2. Define a short command

Go into the project folder and define `gpu-ibs` once in each new terminal. With Docker:

```bash
cd my_project
gpu-ibs () { docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$PWD":/data ahmadshakleya/gpu-ibs:latest gpu-ibs "$@"; }
```

With Apptainer or Singularity:

```bash
cd my_project
gpu-ibs () { apptainer exec --nv /path/to/gpu-ibs.sif gpu-ibs "$@"; }
```

`gpu-ibs` then runs the three tools, `kmc`, and the image's own commands (`selftest`, `example`, `fit`). All the commands below are run from `my_project` with paths relative to it. The Docker container only sees `my_project`, so keep your reads inside it, or mount their folder as well (add `-v /scratch/reads:/reads` to the `docker run` line and use `/reads/...` paths). Apptainer sees your home directory and the current folder; add `--bind /scratch` for data elsewhere.

### 3. Count the k-mers of each sample

```bash
ls reads/S1_*.fastq.gz > S1.reads
mkdir -p kmc_sets/S1 tmp
gpu-ibs kmc -k25 -ci2 -cs255 -t16 -m32 @S1.reads kmc_sets/S1/S1 tmp
```

This writes the database `kmc_sets/S1/S1.kmc_pre` and `kmc_sets/S1/S1.kmc_suf`. Repeat it for every sample. The options:

- `-k25` is the k-mer length. Use the same k for all the samples you want to compare. Up to k = 32 the tools use their fast GPU and CPU code; a larger k works, but runs on the slower original code. IBSpy uses 31 by default, and we used 25.
- `-ci2` ignores k-mers seen only once, which are mostly sequencing errors.
- `-cs255` stores the counts in one byte. The scan only needs to know whether a k-mer is present.
- `-t16 -m32` gives KMC 16 threads and 32 GB of RAM. Set them to what your machine has: if KMC cannot get the memory, it stops with `std::bad_alloc`.
- If your reads are in FASTA rather than FASTQ, add `-fm`.

KMC needs temporary disk space of two to three times the size of the compressed reads, so for large samples put `tmp` on a scratch disk.

### 4. Run the IBS scan

```bash
gpu-ibs fastibs kmc_sets/S1 reference FastIBS_runs 50000
```

The last number is the window size in bases. 50000 is IBSpy's default; it has to be larger than k. The result is `FastIBS_runs/S1_v_MorexV3_50000.tsv` for a reference file called `MorexV3_genome.fasta` (the name is cut at `_genome`; other file names are kept whole).

### 5. Scan many samples at once

Scanned one at a time, every sample reads the reference again. In panel mode the reference is read once and several samples are scanned together. The output files are the same as from separate runs.

```bash
gpu-ibs fastibs kmc_sets/S1 reference FastIBS_runs 50000 --db=kmc_sets/S2 --db=kmc_sets/S3
```

For a long list, put the sample folders in a file, one per line:

```bash
printf 'kmc_sets/%s\n' S2 S3 S4 S5 > panel.txt
gpu-ibs fastibs kmc_sets/S1 reference FastIBS_runs 50000 --db-list=panel.txt
```

If not all the samples fit on the GPU together, the rest are done in further rounds. On a cluster you can also run one job per sample; [`docs/DOCKER.md`](docs/DOCKER.md#apptainer-and-singularity-hpc-clusters) has a SLURM script.

### 6. Per-base coverage and shared k-mers

```bash
gpu-ibs fastibsmapper kmc_sets/S1 reference FastIBS_runs
gpu-ibs KDBIntersect kmc_sets/S1 kmc_sets/S2
```

### Restarting and help

If an output file already exists, the tools skip it. So when a run stops halfway, start the same command again and it continues with what is missing; delete a file to compute it again. Every tool prints all its options with `--help`.


## Output files

`fastibs` writes a tab-separated table per reference file, one row per window:

```
seqname  start   end     total_kmers  observed_kmers  variations  kmer_distance
chr1     0       50000   49976        41450           307         1158
chr1     49975   99975   49976        40905           315         1518
```

| column | meaning |
|---|---|
| `seqname` | the reference sequence (the full FASTA header line) |
| `start`, `end` | the window, counted from 0, with `end` not included. Each window starts k bases before the previous one ends. |
| `total_kmers` | the reference's k-mers in the window (window length − k + 1, fewer near `N`s and at the end of a sequence) |
| `observed_kmers` | how many of those k-mers the sample has |
| `variations` | the number of stretches of reference k-mers missing from the sample, that is, places where the sample differs from the reference |
| `kmer_distance` | IBSpy's IBS distance: the estimated number of differing bases. A single SNP removes k k-mers in a row and counts as 1. |

`fastibsmapper` writes `S1_v_<reference file name>.txt` (without the `.fasta`): for each reference sequence, a line with its name and then one line with a comma-separated count for every base, from 0 to k.

`KDBIntersect` prints one number, the k-mers the two samples share. Messages go to standard error, so `n=$(gpu-ibs KDBIntersect kmc_sets/S1 kmc_sets/S2)` stores just the number.

[`docs/REFERENCE.md`](docs/REFERENCE.md) has the exact rules for all three.


## GPUs

- Any NVIDIA GPU from the V100 onwards works (compute capability 7.0 or newer), with driver 525.60 or newer. We have tested V100 and A100 GPUs.
- The GPU is used automatically. If it is not, the tools print one line saying why and run on the CPU. Use `--cpu` to keep them on the CPU.
- The sample's k-mer database has to fit in GPU memory, at 4 to 8 bytes per k-mer. A barley sample with 2.1 billion k-mers at k = 25 needs 9.3 GB. `gpu-ibs fit kmc_sets/S1` tells you how much a database needs and whether it fits. If it does not fit, the tools run on the CPU.
- With several GPUs, all of them are used and the database is divided between them, so it only has to fit their combined memory: a 96 GB database runs on two 80 GB A100s. This does not make one scan much faster. To get through many samples faster, use panel mode or run one process per GPU (`CUDA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=1`, ...). `--gpus=1` limits a run to one GPU.
- `FASTIBS_THREADS=16` limits the number of CPU threads (with Docker, add `-e FASTIBS_THREADS=16` to the `docker run` line).

If the GPU is not used, the line the tools print tells you why:

- `no NVIDIA driver is loaded`: the machine has no NVIDIA driver, or the container was started without `--gpus all` (Docker) or `--nv` (Apptainer).
- `the driver supports CUDA 11.x`: the driver is too old; it needs to be 525.60 or newer.
- `no CUDA-capable device is detected`: the driver sees no GPU. Check `nvidia-smi` and `CUDA_VISIBLE_DEVICES`.


## Building from source

On Linux, with conda:

```bash
git clone https://github.com/accl-kaust/GPU-IBS.git
cd GPU-IBS
conda create -n gpuibs -c conda-forge -c bioconda cmake ninja boost-cpp zlib kmc -y
conda activate gpuibs
conda install -c conda-forge cuda-toolkit=12.4 -y   # for the GPU version; skip it for CPU only
bash compile.sh
bash scripts/run_tests.sh
```

`compile.sh` puts `fastibs`, `fastibsmapper` and `KDBIntersect` in `bin/`. It builds the GPU version when it finds the CUDA compiler (a GPU is not needed to build), and `FASTIBS_CUDA=off bash compile.sh` builds a CPU-only version. `scripts/run_tests.sh` checks the build against the original FastIBS code, and the GPU against the CPU when a GPU is present. `fastibs --version` shows which version you are running.


## The original FastIBS scripts

FastIBS's own installer and scripts (`install.sh`, `run.sh`, `intersect.sh` and `submit_jobs.sh`) still work as before; [`docs/INSTALL.md`](docs/INSTALL.md) describes them. `run.sh` and `intersect.sh` can also use the ready-made image: `SIF_NAME=gpu-ibs.sif bash run.sh ...`.


## More documentation

- [`docs/REFERENCE.md`](docs/REFERENCE.md): every option, environment variable and output rule.
- [`docs/DOCKER.md`](docs/DOCKER.md): the container image, clusters and a SLURM example.
- [`docs/REALDATA_WALKTHROUGH.md`](docs/REALDATA_WALKTHROUGH.md): a real 100 GB barley sample, from download to results.
- For developers: [`scripts/README.md`](scripts/README.md) (the tests and correctness checks) and [`docs/PHASES.md`](docs/PHASES.md) (notes on how the GPU version was built).
