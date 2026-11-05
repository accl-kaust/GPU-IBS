#!/bin/bash

# Function to display help message
function display_help {
    echo "Usage: $0 <KMCDB1Path> <KMCDB2Path>"
    echo
    echo "This script runs a Singularity/Apptainer container to compute the intersection count."
    echo "  example:        bash $0 data/ECOLI data/G123"
    echo
    echo "The GPU is used automatically when the host has an NVIDIA driver (the container"
    echo "is started with --nv); FASTIBS_GPU=0 in the environment forces the CPU path."
}

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    display_help
    exit 1
fi

# Assign arguments to variables
KMCDB1=$1
KMCDB2=$2

# Fixed values
project_dir=$(pwd)
# The settings install.sh used (SIF_NAME, CONTAINER_CMD), when run from the project root
if [ -f init/config.sh ]; then source init/config.sh; fi
singularity_image="${SIF_NAME:-fastibs.sif}"

# Container runtime: apptainer when installed, else singularity
container_cmd=${CONTAINER_CMD:-$(command -v apptainer >/dev/null 2>&1 && echo apptainer || echo singularity)}
# GPU access: --nv when the host has an NVIDIA driver, unless FASTIBS_GPU=0
nv_flag=""
if [ "${FASTIBS_GPU:-1}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1; then
    nv_flag="--nv"
fi

if [ ! -f "$singularity_image" ]; then
    echo "Error: Singularity image '$singularity_image' does not exist."
    exit 1
fi

# The self-contained image (the Dockerfile's) has KDBIntersect on its PATH;
# fastibs.sif of install.sh runs the project's bin/ through the /project bind.
if ${container_cmd} exec "${singularity_image}" sh -c 'command -v KDBIntersect' > /dev/null 2>&1; then
    tool=KDBIntersect
else
    tool=/project/bin/KDBIntersect
fi

# Run the container with the bound directories and the specified DB paths
# (argument order as in the original script; KDBIntersect picks the smaller
# database for random access itself, so the count does not depend on it)
${container_cmd} exec ${nv_flag} --bind "${project_dir}":/project "${singularity_image}" \
    "${tool}" "${KMCDB2}" "${KMCDB1}"
