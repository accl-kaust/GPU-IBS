# The original FastIBS installer and scripts

GPU-IBS keeps FastIBS's own way of deploying the tools: `install.sh` builds a toolchain image, compiles the tools inside it into `./bin` of your clone, and packages a small runtime image as `fastibs.sif`; `run.sh`, `intersect.sh` and `submit_jobs.sh` then run the tools through that `.sif`, with the project folder mounted at `/project`. Everything below works as in FastIBS, with the GPU path added.

For new installations the self-contained container image is simpler: nothing is compiled on the host, KMC is included, and the tools are on `PATH` ([DOCKER.md](DOCKER.md)). A native build is in the [README](../README.md#build-from-source).

## Requirements

- **Docker** (to build the images and compile the binaries), *or* Apptainer/Singularity alone (see below)
- **Singularity** or **Apptainer** (for generating and running the `.sif` image)
- **Bash**
- **KMC**, to build the k-mer databases the tools read. This flow does not ship it: install it yourself (`conda install -c bioconda kmc`, or from [refresh-bio/KMC](https://github.com/refresh-bio/KMC)); the container image of [DOCKER.md](DOCKER.md) includes it. The repository vendors only KMC's database-reading API (`KMC/kmc_api`), which CMake compiles into the tools.
- For GPU execution: an NVIDIA GPU of compute capability 7.0, 8.0 or 9.0 with a driver ≥ 525.60 (CUDA 12 minor-version compatibility; the binaries were built with CUDA 12.4 and validated on drivers 535 and 570). Code for V100, A100 and H100 is compiled in; the validation runs used **V100 and A100** GPUs, and H100 is built for but not yet validated. Inside Docker, the NVIDIA Container Toolkit is needed for `--gpus all`; Singularity/Apptainer use `--nv`.

> ⚠️ Note: You do *not* need to install CMake, Ninja or the CUDA toolkit on your host system — they are included in the toolchain image.

Without a GPU everything still works: the same binaries take the CPU path (`fastibs` and `fastibsmapper` report `[gpu] no CUDA device found, using CPU path (cudaGetDeviceCount: <reason>)`, `KDBIntersect` reports `[gpu] no CUDA device found, using CPU intersection (cudaGetDeviceCount: <reason>)`) and produce the same results. With `--gpu` and no usable device, `fastibs` and `fastibsmapper` also print a warning and still finish on the CPU.

---


## Build and install

### 1. Clone the repository

```bash
mkdir <project-folder>
cd <project-folder>
git clone https://github.com/accl-kaust/GPU-IBS.git .
```

### 2. (Optional) Edit Configuration

Edit `init/config.sh` if you want to customize the image and container names, or the GPU build:

```bash
IMAGE_NAME=fastibs_img        # Docker image name (runtime image; fastibs.sif is built from it)
CONTAINER_NAME=fastibs_cont   # Docker container name (the compile container)
FASTIBS_CUDA=auto             # auto | on | off  -- off builds CPU-only binaries and images
CUDA_ARCHS="70;80;90"         # GPU architectures compiled in (V100, A100, H100)
```

### 3. Run the Installer

```bash
bash install.sh
```

This script performs the same steps as the original FastIBS installer:

- Builds the Docker images: a **toolchain** image (`${IMAGE_NAME}_build`, CUDA 12.4 + g++ + CMake + Ninja) and a small **runtime** image (`${IMAGE_NAME}`, Ubuntu 22.04 + zlib)
- Starts the toolchain container with your project directory mounted at `/project` (with `--gpus all` when the host has the NVIDIA runtime)
- Compiles the C++ binaries inside the container (`compile.sh`, run as your user), so they appear in `./bin` on the host, owned by you
- Saves the runtime image as a `.tar` archive
- Builds `fastibs.sif` from the `.tar` with `apptainer` (or `singularity`, whichever is installed)
- Removes the intermediate `.tar` file

The CUDA runtime is linked statically into the binaries, so the runtime image and `fastibs.sif` stay small; the GPU driver library is provided by the host at run time. (In the Docker-less flow below, `fastibs.sif` is the toolchain image itself and correspondingly larger.)

**Without Docker** (typical on HPC login nodes): if Docker is missing or its daemon is not usable, `install.sh` builds `fastibs.sif` directly from `init/fastibs.def` (a toolchain image, `--fakeroot` when not root) and compiles the binaries inside it with a clean environment, so an active conda environment on the login node does not influence the build. The same two steps by hand:

```bash
apptainer build --fakeroot fastibs.sif init/fastibs.def
apptainer exec --cleanenv --bind .:/project fastibs.sif bash -c "cd /project && bash compile.sh"
```


## Output binaries

After a successful build, the following executables will be available in `/project/bin`:

- `fastibs`
- `fastibsmapper`
- `KDBIntersect`
- `probe_bench` (GPU builds only: the probe-throughput microbenchmark used by the validation gates)


## The compile container (Docker)

To manually enter the running compile container:

```bash
docker exec -it <container_name> bash
```

Or, using variables from the config:

```bash
source init/config.sh
docker exec -it ${CONTAINER_NAME} bash
```

The container sees the project at `/project`; to run the GPU path inside Docker the container must have been started with the GPUs passed through (`start_cont.sh` does this when it detects the NVIDIA runtime). The Docker flow is kept from FastIBS and builds, but no Docker run is among the recorded validations (those used the Apptainer image on a V100 node and native builds on V100 and A100 nodes).


## Running through fastibs.sif (Singularity / Apptainer)

Once `fastibs.sif` is built, you can test that the FastIBS tools are properly installed as follows (replace `apptainer` by `singularity` if that is what your system provides; the flags are the same):

```bash
apptainer exec --bind .:/project fastibs.sif /project/bin/fastibs --help
apptainer exec --bind .:/project fastibs.sif /project/bin/fastibsmapper --help
apptainer exec --bind .:/project fastibs.sif /project/bin/KDBIntersect --help
```

Add `--nv` to give the container access to the host's GPUs; without it (or without a GPU) the tools take the CPU path:

```bash
apptainer exec --nv --bind .:/project,/path/to/FastIBSData:/project/data fastibs.sif \
    /project/bin/fastibs /project/data/kmc_sets/ECOLI /project/data/reference /project/data/FastIBS_runs 50000
```

`run.sh` and `intersect.sh` add `--nv` automatically when the host has an NVIDIA driver.

To check the packaged image on a GPU node as a batch job (interactive `srun` allocations can wait a long time for a free GPU), `scripts/slurm_sif_gate.sbatch` runs `fastibs` inside the image on the GPU and on the CPU (`--cpu`) and requires the two outputs to be byte-identical; it also compares against a native `build/fastibs` when one runs on the host. Paths are relative to the project root, which is bind-mounted at `/project`:

```bash
sbatch -x <unhealthy nodes> scripts/slurm_sif_gate.sbatch corpus_big/db corpus_big/refs 50000   # result in sif_<jobid>.out
```

If you want to use the tools do not forget to mount a data volume with a structure similar to the following:

```bash
FastIBSData/ # data volume root
├── FastIBS_runs # results folder (initially empty)
│   ├── ECOLI_v_ecoli_50000.tsv # fastibs result file
│   ├── ECOLI_v_ecoli_genome.txt # fastibs mapper result file
│   └── ECOLI_v_TA1675_50000.tsv
├── kmc_sets # KMC database (A path to your KMC KBs)
│   ├── BW_01002
│   │   ├── BW_01002.res.kmc_pre
│   │   └── BW_01002.res.kmc_suf
│   ├── ECOLI
│   │   ├── ecoli.res.kmc_pre
│   │   └── ecoli.res.kmc_suf
│   └── IG90747
│       ├── IG90747.res.kmc_pre
│       └── IG90747.res.kmc_suf
└── reference # reference database
    ├── ecoli_genome.fasta
    └── TA1675_genome.fasta
```


## Clean build

To remove all build artifacts and recompile from scratch:

```bash
rm -rf build bin
bash compile.sh
```

Or re-run the full installation pipeline:

```bash
bash install.sh
```


## Running on HPC environments: submit_jobs.sh

A bash script `submit_jobs.sh` is provided as an example. This script is used to automate the generation and submission of multiple SLURM jobs based on a list of accessions. Each job will execute the FastIBS program with a specified window size for a given accession.

### 1. Input Arguments
   - **`accessions_file`**: A text file containing a list of accession identifiers. Each accession corresponds to a KMC database and a separate job that will be processed.
   - **`data_path`**: The path to the directory containing the required data for the FastIBS program. This is typically the data associated with the accessions, including as detailed earlier folders for `kmc_sets`, `reference`, and possibly `results`
   - **`window_size`**: The window size parameter that will be passed to the FastIBS program. This specifies the size of the sequence windows used for calculating IBS (Identity by State) distances.

### 2. Functionality
   - The script first checks if the correct number of arguments is provided.
   - It checks if the provided accessions file exists and is readable.
   - For each accession listed in the file, a SLURM job script is dynamically generated. This job script sets job parameters, such as memory allocation, CPU cores, and runtime limits. It also calls the `run.sh` script to process the FastIBS program for that particular accession.
   - Once the job script is generated for each accession, the script can be configured to submit each job to the SLURM job scheduler for parallel execution. (The submission line is currently commented out for safety, but can be enabled by uncommenting the line `sbatch $job_script`.)
   - Setting `GPU_GRES` (for example `GPU_GRES=gpu:a100:1 bash submit_jobs.sh ...`) adds a matching `#SBATCH --gres` line to every job; `run.sh` then starts the container with `--nv` and `fastibs` uses the GPU. Without it the generated scripts are identical to the original FastIBS ones.

```bash
# Loop over each accession in the file
while IFS= read -r accession
do
    # Generate a SLURM job script
    job_script="job_${accession}.sh"
    cat << EOF > $job_script
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=FastIBS_${accession}
#SBATCH --output=FastIBS_${accession}.out
#SBATCH --error=FastIBS_${accession}.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=50
${gres_line}bash run.sh ${accession} ${data_path} ${window_size}
EOF

    # Submit the job - uncomment below to submit
    #sbatch $job_script
done < "$accessions_file"
```

### 3. Job Configuration
   - Each SLURM job is given unique names for job output and error logs (`FastIBS_${accession}.out` and `FastIBS_${accession}.err`).
   - Resources are allocated based on the job requirements: 24 hours of runtime, 256GB of memory, and 50 CPUs per task. Change these according to your requirements; a GPU job needs far fewer cores (the validation gates use 16) and `--gres` for the device.
   - Jobs are submitted in parallel, each handling a different accession, speeding up processing for large datasets.
   - Within a job the reference is processed in parallel by the tool's thread pool: on the CPU path one task per window; on the GPU path the windows are batched and streamed through the device while the pool packs the sequence and computes the window statistics.

### Example Use Case
To run multiple FastIBS jobs in parallel, assuming you have a list of accessions (`accessions.txt`), data directory (`./FastIBSData/`), and a desired window size of 50,000:

```bash
bash submit_jobs.sh accessions.txt ./FastIBSData/ 50000
```
