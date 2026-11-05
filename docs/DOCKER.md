# The GPU-IBS container image

One image holds everything needed to go from sequencing reads to IBS tables:

| in the image | what it is |
|---|---|
| `fastibs`, `fastibsmapper`, `KDBIntersect` | the GPU-IBS tools, built with the GPU path for V100, A100 and H100 (compute capability 7.0, 8.0, 9.0, plus PTX for newer GPUs); with no GPU visible they run the CPU path and give the same output |
| `kmc`, `kmc_tools`, `kmc_dump` | [KMC](https://github.com/refresh-bio/KMC) 3.2.4 (the official static release, checked against its sha256), to count the k-mers of each sample |
| `gpu-ibs` | the image's helper: `gpu-ibs info`, `gpu-ibs selftest`, `gpu-ibs example`, `gpu-ibs fit <kmc_db>` (GPU memory a database needs) |
| `probe_bench` | the probe microbenchmark of the validation gates |

The image is `linux/amd64`, based on Ubuntu 22.04: a 42 MB download, 161 MB unpacked. The CUDA runtime is linked into the tools; the GPU driver comes from the host.

## Get the image

```bash
docker pull ahmadshakleya/gpu-ibs:latest
```

`latest` follows the main branch. For an analysis you want to reproduce later, pull a release instead, for example `ahmadshakleya/gpu-ibs:1.0.0` (each release `vX.Y.Z` is published as `X.Y.Z` and `X.Y`, and every build of main as `sha-<commit>`); `gpu-ibs info` and `fastibs --version` say which one you have. The image is on [Docker Hub](https://hub.docker.com/r/ahmadshakleya/gpu-ibs); the same tags are also published to GitHub's registry as `ghcr.io/accl-kaust/gpu-ibs`.

Or build it from a clone of this repository (no GPU needed; the first build downloads the 4 GB CUDA build image, then compiles for a few minutes):

```bash
git clone https://github.com/accl-kaust/GPU-IBS.git
cd GPU-IBS
docker build -t gpu-ibs .
```

Build options (`--build-arg NAME=value`):

| argument | default | meaning |
|---|---|---|
| `FASTIBS_CUDA` | `ON` | `OFF` builds CPU-only tools (`docker build -t gpu-ibs:cpu --build-arg FASTIBS_CUDA=OFF .`) |
| `CUDA_ARCHS` | `70;80;90` | GPU architectures compiled in |
| `CUDA_VERSION` | `12.4.1` | CUDA toolkit of the build stage (`nvidia/cuda:<version>-devel-ubuntu22.04`) |
| `BUILD_JOBS` | all CPUs | parallel compile jobs; set it (e.g. `4`) on a machine with little memory, since each GPU compile job needs about 2 GB |
| `VERSION`, `REVISION` | `dev`, `unknown` | written into the image's labels and shown by `gpu-ibs info` |

`docker build --target test .` builds the tools and the regression tests and runs the whole host test suite (`scripts/run_tests.sh`) inside the build; the build fails if a test fails.

The commands below write `gpu-ibs` for the image name; use `ahmadshakleya/gpu-ibs:latest` if you pulled it.

## Check it on your machine

```bash
docker run --rm --gpus all gpu-ibs gpu-ibs selftest
```

The selftest makes a small synthetic data set (a 2.2 Mbp reference with lowercase bases, N runs and contigs shorter than k, and three samples), runs the three tools on it and checks that

- the IBS table, the coverage file and the intersection count equal the recorded reference results,
- the GPU output equals the CPU output, byte for byte, when a GPU is visible,
- the CPU output equals that of the original FastIBS code.

It prints which GPU it used, or why it ran on the CPU, and ends with `SELFTEST PASSED`. Add `--require-gpu` to make a run without a GPU a failure (useful on a GPU node of a cluster), `--keep <dir>` to keep the data and logs.

`gpu-ibs info` prints the versions, the build, and the CPUs and GPUs the container sees.

## Run it on your data

The image's working directory is `/data`. Mount your project folder there and run as yourself, so the results belong to you and relative paths work:

```bash
docker run --rm --gpus all -u $(id -u):$(id -g) -v "$PWD":/data gpu-ibs \
    fastibs kmc_sets/SAMPLE1 reference FastIBS_runs 50000
```

A shell function saves typing (put it in `~/.bashrc`):

```bash
gpu-ibs () { docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$PWD":/data gpu-ibs "$@"; }

gpu-ibs kmc -k25 -ci2 -cs255 -t16 -m32 @SAMPLE1.reads kmc_sets/SAMPLE1/SAMPLE1 tmp
gpu-ibs fastibs kmc_sets/SAMPLE1 reference FastIBS_runs 50000
gpu-ibs KDBIntersect kmc_sets/SAMPLE1 kmc_sets/SAMPLE2
```

Only the mounted folder is visible inside the container: every path must lie under it (or add more `-v host_path:container_path` mounts). `gpu-ibs example` writes a small worked project into the current folder and runs every tool on it, printing each command. `gpu-ibs fit kmc_sets/SAMPLE1` says how much GPU memory a database needs and whether it fits the visible GPU (`-e SHARDS=2` for the index split across two GPUs).

**Without a GPU** drop `--gpus all`; the tools run on the CPU and write the same files. **Threads:** the CPU path uses every CPU of the machine by default; `-e FASTIBS_THREADS=16` (or `--cpus 16`) limits it. All environment switches of the tools ([`REFERENCE.md`](REFERENCE.md#environment-variables)) are passed with `-e NAME=value`.

## GPUs in Docker

`--gpus all` needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host and an NVIDIA driver ≥ 525.60 (CUDA 12 minor-version compatibility). `--gpus 1` or `--gpus '"device=1,2"'` select devices; with several GPUs visible the tools split the k-mer index across them (`--gpus=N` limits the count). If the selftest reports the CPU path although the host has a GPU, the reason is printed with it: `no NVIDIA driver is loaded` means the container was started without `--gpus all` (or the NVIDIA Container Toolkit is missing), `the driver supports CUDA 11.x` means the host driver is older than 525.60, `no CUDA-capable device is detected` means the driver sees no GPU.

## Apptainer and Singularity (HPC clusters)

Clusters rarely allow Docker, but Apptainer and Singularity run the same image without root:

```bash
apptainer pull gpu-ibs.sif docker://ahmadshakleya/gpu-ibs:latest
apptainer exec --nv gpu-ibs.sif gpu-ibs selftest --require-gpu
apptainer exec --nv gpu-ibs.sif fastibs kmc_sets/SAMPLE1 reference FastIBS_runs 50000
```

(`singularity` takes the same commands.) `--nv` makes the host's GPUs and driver visible; without it the tools run on the CPU. Apptainer mounts your home and the current folder by default; add `--bind /scratch/project` for data elsewhere. To convert a locally built image instead of pulling one, save it and build the `.sif` from the archive:

```bash
docker save gpu-ibs -o gpu-ibs.tar
apptainer build gpu-ibs.sif docker-archive://gpu-ibs.tar
```

A SLURM job then looks like this (one sample per job; resources are an example):

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
export FASTIBS_THREADS=$SLURM_CPUS_PER_TASK
apptainer exec --nv gpu-ibs.sif fastibs kmc_sets/$1 reference FastIBS_runs 50000
```

`--mem` must hold one copy of the sample's database (the CPU path keeps it in memory, the GPU path reads it once to upload it; `gpu-ibs fit` prints the figure) plus the largest reference file, which is read into memory whole.

## What has been checked

- With Docker 29 on Ubuntu (WSL2), no GPU: `docker build`, `docker build --target test .` (43 checks: the regression tests and the tools against the original FastIBS code), `gpu-ibs selftest`, `gpu-ibs example`, `gpu-ibs fit`, and a run from gzipped FASTQ reads through `kmc` to a `fastibs` table and `KDBIntersect`.
- On a KAUST Ibex node with a V100-SXM2-32GB (driver 570.86.15), the image converted with `singularity build gpu-ibs.sif docker-archive://...` (Singularity 3.9.7, no root, 9 s): `gpu-ibs selftest --require-gpu` (GPU output = CPU output = original code = the recorded results), `gpu-ibs selftest` without `--nv` (CPU path), `gpu-ibs example` (every tool on the GPU), the container gate `SIF=gpu-ibs.sif scripts/slurm_sif_gate.sbatch` (GPU output = CPU output = native build), FastIBS's `run.sh` and `intersect.sh` with `SIF_NAME=gpu-ibs.sif`, and a real barley sample (ERR6098905, 2.06×10⁹ k-mers at k = 25) against chromosome 1H of MorexV3 on the GPU inside the container, whose table is byte-identical to that of a native CPU build.
- Not yet run: `docker run --gpus all` itself on a GPU host (no machine with both Docker and a GPU was available); the image follows the NVIDIA Container Toolkit's conventions (`NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`), and through Apptainer/Singularity it has been run on a GPU as above.

## The original FastIBS container flow

`install.sh` keeps FastIBS's own deployment: a toolchain image compiles the tools into `./bin` of your clone, and `fastibs.sif` is made from a small runtime image; `run.sh`, `intersect.sh` and `submit_jobs.sh` then run the tools through that `.sif` with the project mounted at `/project`. It is described in the main [README](../README.md#the-original-fastibs-installer). The image of this page needs none of it.
