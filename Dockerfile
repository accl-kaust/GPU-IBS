# GPU-IBS container image: fastibs, fastibsmapper and KDBIntersect with the
# GPU path compiled in (they fall back to the CPU when no GPU is visible),
# plus KMC to build the k-mer databases from reads. Self-contained: nothing
# is compiled on the host and nothing has to be mounted except your data.
#
#   docker build -t gpu-ibs .                                   # GPU + CPU image
#   docker build -t gpu-ibs:cpu --build-arg FASTIBS_CUDA=OFF .  # CPU-only binaries
#   docker build --target test .                                # build + whole host test suite
#
#   docker run --rm --gpus all gpu-ibs gpu-ibs selftest
#   docker run --rm --gpus all -u $(id -u):$(id -g) -v "$PWD":/data gpu-ibs \
#       fastibs kmc_sets/SAMPLE reference FastIBS_runs 50000
#
# docs/DOCKER.md has the details (Apptainer/Singularity, HPC, GPU drivers).
# init/Dockerfile is the original FastIBS compile-container flow (install.sh).

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG FASTIBS_CUDA=ON
ARG CUDA_ARCHS="70;80;90"
ARG KMC_VERSION=3.2.4
ARG KMC_SHA256=158f2084f8d928b3f33b8aaf7d1220fee4183bf46837787e5e6b16bbdf54d31d

# ---------------------------------------------------------------- build ----
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake ninja-build g++ zlib1g-dev libboost-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src/gpu-ibs
COPY src/ src/
COPY KMC/ KMC/
ARG FASTIBS_CUDA
ARG CUDA_ARCHS
# BUILD_JOBS limits the parallel compile jobs (nvcc needs ~2 GB each)
ARG BUILD_JOBS=
RUN set -e; \
    archs=""; \
    if [ "${FASTIBS_CUDA}" = ON ]; then archs="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"; fi; \
    cmake -S src -B build -G Ninja -DFASTIBS_CUDA="${FASTIBS_CUDA}" ${archs:+"$archs"}; \
    cmake --build build --parallel ${BUILD_JOBS}; \
    mkdir -p /opt/gpu-ibs/bin; \
    cp build/fastibs build/fastibsmapper build/KDBIntersect /opt/gpu-ibs/bin/; \
    if [ -x build/probe_bench ]; then cp build/probe_bench /opt/gpu-ibs/bin/; fi; \
    strip /opt/gpu-ibs/bin/*

# ------------------------------------------------------------------ kmc ----
# The official static KMC release, checked against its sha256.
FROM ubuntu:${UBUNTU_VERSION} AS kmc
ARG DEBIAN_FRONTEND=noninteractive
ARG KMC_VERSION
ARG KMC_SHA256
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN set -e; \
    curl -fsSL -o /tmp/kmc.tar.gz \
        "https://github.com/refresh-bio/KMC/releases/download/v${KMC_VERSION}/KMC${KMC_VERSION}.linux.x64.tar.gz"; \
    echo "${KMC_SHA256}  /tmp/kmc.tar.gz" | sha256sum -c -; \
    mkdir -p /tmp/kmc /opt/kmc/bin; \
    tar -xzf /tmp/kmc.tar.gz -C /tmp/kmc; \
    cp /tmp/kmc/bin/kmc /tmp/kmc/bin/kmc_tools /tmp/kmc/bin/kmc_dump /opt/kmc/bin/; \
    rm -rf /tmp/kmc /tmp/kmc.tar.gz

# ----------------------------------------------------------------- test ----
# docker build --target test . : builds the regression tests and runs them,
# then checks the tools against the original FastIBS code path (no GPU here:
# the GPU build takes its CPU fallback, which is part of what is checked).
FROM build AS test
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends gawk \
 && rm -rf /var/lib/apt/lists/*
COPY --from=kmc /opt/kmc/bin/ /usr/local/bin/
COPY tests/ tests/
COPY scripts/run_tests.sh scripts/make_test_corpus.sh scripts/make_test_panel.sh scripts/gate_multidb.sh scripts/
RUN BIN_DIR=/opt/gpu-ibs/bin bash scripts/run_tests.sh /tmp/test_work

# -------------------------------------------------------------- runtime ----
FROM ubuntu:${UBUNTU_VERSION} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
# gawk: the synthetic data of `gpu-ibs selftest` (another awk gives other data)
RUN apt-get update \
 && apt-get install -y --no-install-recommends gawk zlib1g \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/gpu-ibs/bin/ /usr/local/bin/
COPY --from=kmc /opt/kmc/bin/ /usr/local/bin/
COPY scripts/make_test_corpus.sh scripts/make_test_panel.sh scripts/realdata/03_preflight.sh /usr/local/share/gpu-ibs/scripts/
COPY docker/gpu-ibs /usr/local/bin/gpu-ibs
COPY LICENSE /usr/local/share/doc/gpu-ibs/LICENSE
ARG FASTIBS_CUDA
ARG CUDA_ARCHS
ARG CUDA_VERSION
ARG KMC_VERSION
ARG VERSION=dev
ARG REVISION=unknown
RUN chmod 0755 /usr/local/bin/gpu-ibs \
 && printf 'version=%s\nrevision=%s\ncuda=%s\ncuda_version=%s\ncuda_archs=%s\nkmc=%s\n' \
        "${VERSION}" "${REVISION}" "${FASTIBS_CUDA}" "${CUDA_VERSION}" "${CUDA_ARCHS}" "${KMC_VERSION}" \
        > /usr/local/share/gpu-ibs/build-info
# The NVIDIA Container Toolkit injects the host's driver library (libcuda)
# and nvidia-smi into the container for these capabilities; the CUDA runtime
# itself is linked statically into the tools.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LC_ALL=C.UTF-8
LABEL org.opencontainers.image.title="GPU-IBS" \
      org.opencontainers.image.description="IBS distances from KMC k-mer databases (FastIBS) on NVIDIA GPUs and CPUs, with KMC" \
      org.opencontainers.image.source="https://github.com/accl-kaust/GPU-IBS" \
      org.opencontainers.image.licenses="GPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
WORKDIR /data
CMD ["gpu-ibs"]
