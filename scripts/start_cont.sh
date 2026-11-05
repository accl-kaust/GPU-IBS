#!/bin/bash
# Starts the compile container (toolchain image) with the project mounted at
# /project. Run from init/ (install.sh does). When the host has an NVIDIA
# driver and Docker's NVIDIA runtime, the GPUs are passed through so the GPU
# binaries can also be run from inside the container; compiling needs none.
set -e

# Read config values
source config.sh

echo "##############################"
echo "Starting container ${CONTAINER_NAME}"

# Check if the container exists and remove it if it does
if docker container ls -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} exists, removing..."
    docker container stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker container rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Container ${CONTAINER_NAME} removed!"
fi

# Get the absolute path of the project directory
project_dir=$(realpath ..)

gpu_flag=()
# Probe the capability itself (the toolchain image exists: build_img.sh ran
# first) rather than grepping `docker info`, which can mention "nvidia" for
# unrelated reasons or say nothing although --gpus works.
if command -v nvidia-smi >/dev/null 2>&1 \
   && docker run --rm --gpus all "${BUILDER_IMAGE_NAME}" true >/dev/null 2>&1; then
    gpu_flag=(--gpus all)
    echo "NVIDIA container toolkit detected: passing GPUs through (--gpus all)"
fi

# Run a new container instance
# You may use --user $(id -u):$(id -g) to run under user id instead of root
container_id=$(docker run -d --name "${CONTAINER_NAME}" -it "${gpu_flag[@]}" \
    -v "${project_dir}":/project -p 8087:8087 "${BUILDER_IMAGE_NAME}")
echo "New container instance ${CONTAINER_NAME} is running with ID: ${container_id}"

echo "##############################"
