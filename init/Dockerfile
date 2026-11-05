# GPU-IBS container images (same flow as the original FastIBS: the project
# directory is bind-mounted at /project and the binaries are compiled inside
# the container, so they land in ./bin on the host).
#
# Two stages, built by scripts/build_img.sh:
#   builder : CUDA toolkit + C++ toolchain. install.sh starts this image as
#             ${CONTAINER_NAME} and runs compile.sh in it.
#   runtime : what fastibs.sif is made from -- just the shared libraries the
#             binaries need (libstdc++, zlib). The CUDA runtime is linked
#             statically into the binaries; the GPU driver library comes from
#             the host at run time (singularity/apptainer --nv, docker --gpus).
#
# CPU-only: FASTIBS_CUDA=off in init/config.sh switches BUILDER_BASE to
# ubuntu:22.04 and the binaries have no GPU path. The default CUDA build also
# runs on machines without a GPU (it reports "[gpu] no CUDA device found" and
# takes the CPU path), so one image serves both kinds of node.
ARG BUILDER_BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04
ARG RUNTIME_BASE=ubuntu:22.04

FROM ${BUILDER_BASE} AS builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential g++ gdb \
        cmake ninja-build ccache \
        zlib1g-dev libboost-dev \
        git ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /project/data
WORKDIR /project

FROM ${RUNTIME_BASE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends zlib1g libstdc++6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /project/data
WORKDIR /project
