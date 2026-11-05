#!/bin/bash
# CMake build of the three tools, installed into ./bin (plus probe_bench for
# GPU builds). Works inside the compile container, where the project is
# mounted at /project as in the original FastIBS flow, and natively.
#
#   FASTIBS_CUDA=auto|on|off   GPU path: auto (default) builds it when nvcc is found
#   CUDA_ARCHS="70;80;90"      CUDA architectures of the GPU build (V100, A100, H100)
#
# Ninja is used when available (it is in the container), Make otherwise.
set -e
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

FASTIBS_CUDA=${FASTIBS_CUDA:-auto}
CUDA_ARCHS=${CUDA_ARCHS:-70;80;90}

rm -rf build bin

cuda=OFF
case "${FASTIBS_CUDA}" in
    on)  cuda=ON ;;
    off) cuda=OFF ;;
    *)   if command -v nvcc >/dev/null 2>&1; then cuda=ON; else cuda=OFF; fi ;;
esac
if [ "${cuda}" = ON ]; then
    echo "Building with the GPU path (CUDA architectures ${CUDA_ARCHS})"
else
    echo "Building CPU-only (no nvcc found or FASTIBS_CUDA=off)"
fi

cmake_args=(-S src -B build -DFASTIBS_CUDA=${cuda})
if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
fi
if [ "${cuda}" = ON ]; then
    cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}")
fi
# A conda environment supplies Boost/zlib/nvcc for native builds; inside a
# container the image's own packages are the right ones, even if a host
# conda env leaked into the environment.
if [ -n "${CONDA_PREFIX:-}" ] && [ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]; then
    cmake_args+=(-DCMAKE_PREFIX_PATH="${CONDA_PREFIX}")
fi

cmake "${cmake_args[@]}"
cmake --build build --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

mkdir -p bin
cp build/fastibs build/fastibsmapper build/KDBIntersect bin/
if [ -x build/probe_bench ]; then
    cp build/probe_bench bin/
fi

echo "Build completed successfully."
