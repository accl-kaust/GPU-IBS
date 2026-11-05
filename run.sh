#!/bin/bash

# Function to display help message
function display_help {
    echo "Usage: $0 <KMC_accession> <data_dir> <window_size> [fastibs options]"
    echo
    echo "This script runs a Singularity/Apptainer container with specified directories and parameters."
    echo
    echo "Arguments:"
    echo "  KMC_accession   Accession of the KMC set (folder name under <data_dir>/kmc_sets)"
    echo "  data_dir        Path to the data directory on the host (where kmc_sets and reference folders reside)"
    echo "  window_size     Window size for the FastIBS program"
    echo "  fastibs options Optional, passed through to fastibs (e.g. --gpus=2, --cpu; see fastibs --help)"
    echo "  example:        bash $0 ECOLI FastIBSData 50000"
    echo
    echo "The GPU is used automatically when the host has an NVIDIA driver (the container"
    echo "is started with --nv); FASTIBS_GPU=0 in the environment forces the CPU path."
}

# Check if the correct number of arguments are provided
if [ "$#" -lt 3 ]; then
    display_help
    exit 1
fi

# Assign arguments to variables
KMC_dir=$1
data_dir=$2
window_size=$3
shift 3

# Fixed values
project_dir=$(pwd)
# The settings install.sh used (SIF_NAME, CONTAINER_CMD), when run from the project root
if [ -f init/config.sh ]; then source init/config.sh; fi
singularity_image="${SIF_NAME:-fastibs.sif}"
source_path="/project/data/kmc_sets/${KMC_dir}"
reference_path="/project/data/reference"
results_folder="/project/data/FastIBS_runs"

# Container runtime: apptainer when installed, else singularity
container_cmd=${CONTAINER_CMD:-$(command -v apptainer >/dev/null 2>&1 && echo apptainer || echo singularity)}
# GPU access: --nv when the host has an NVIDIA driver, unless FASTIBS_GPU=0
nv_flag=""
if [ "${FASTIBS_GPU:-1}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1; then
    nv_flag="--nv"
fi

# Check if the directories and the Singularity image exist
if [ ! -d "$data_dir" ]; then
    echo "Error: Data directory '$data_dir' does not exist."
    exit 1
fi

if [ ! -f "$singularity_image" ]; then
    echo "Error: Singularity image '$singularity_image' does not exist."
    exit 1
fi

# The self-contained image (the Dockerfile's, e.g. apptainer pull gpu-ibs.sif
# docker://ahmadshakleya/gpu-ibs) has fastibs on its PATH; fastibs.sif
# of install.sh runs the project's bin/fastibs through the /project bind.
if ${container_cmd} exec "${singularity_image}" sh -c 'command -v fastibs' > /dev/null 2>&1; then
    tool=fastibs
else
    tool=/project/bin/fastibs
fi

# Run the container with the bound directories and the parameters
${container_cmd} exec ${nv_flag} --bind "${project_dir}":/project,"${data_dir}":/project/data "${singularity_image}" \
    "${tool}" "${source_path}" "${reference_path}" "${results_folder}" "${window_size}" "$@"
