#!/bin/bash
# Real-data walkthrough, step 0: toolchain + both builds.
#
#   * conda env ${CONDA_ENV} (cmake, ninja, boost, zlib, kmc, aria2, g++ 12,
#     CUDA 12.4 toolkit) -- created only if it does not exist yet
#   * GPU-IBS: native GPU build of this checkout (compile.sh -> build/, bin/)
#   * original FastIBS: cloned from github.com/githubcbrc/FastIBS into
#     ${ORIG_DIR} and built natively with the same toolchain, so that the
#     baseline is the upstream code, not our CPU path. Upstream's CMake
#     hard-codes /project/KMC/bin/libkmc_core.a (its Docker mount) and asks
#     for CMake 3.0, which CMake 4 refuses; both are patched in place (the
#     library path to the upstream clone's own KMC/bin/libkmc_core.a; this
#     repository ships no KMC binaries).
#
# Usage (login node or a build allocation; no GPU needed to build):
#   bash scripts/realdata/00_setup.sh
# Environment (all optional):
#   CONDA_ENV=gpuibs   ORIG_DIR=<parent-of-checkout>/FastIBS-orig
#   CUDA_ARCHS="70;80;90"   SKIP_ORIG=1 (do not build the original)
set -euo pipefail

GPUIBS_DIR=${GPUIBS_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}
ORIG_DIR=${ORIG_DIR:-$(dirname "${GPUIBS_DIR}")/FastIBS-orig}
CONDA_ENV=${CONDA_ENV:-gpuibs}
CUDA_ARCHS=${CUDA_ARCHS:-70;80;90}
ORIG_URL=${ORIG_URL:-https://github.com/githubcbrc/FastIBS.git}

echo "GPU-IBS checkout : ${GPUIBS_DIR}"
echo "original FastIBS : ${ORIG_DIR}"
echo "conda env        : ${CONDA_ENV}"
echo

# ---- 1. conda environment ----------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not on PATH (install Miniforge, then re-run)" >&2
    exit 1
fi
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "conda env ${CONDA_ENV} exists; not recreating"
else
    echo "creating conda env ${CONDA_ENV} ..."
    conda create -y -n "${CONDA_ENV}" -c conda-forge -c bioconda \
        cmake ninja boost-cpp zlib kmc aria2 gxx=12 gcc=12
    conda install -y -n "${CONDA_ENV}" -c conda-forge cuda-toolkit=12.4
fi
set +u
conda activate "${CONDA_ENV}"
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "toolchain: $(g++ --version | head -1); $(nvcc --version | grep release || echo 'nvcc missing')"
echo

# ---- 2. GPU-IBS native build ---------------------------------------------------
echo "=== building GPU-IBS (FASTIBS_CUDA=on, archs ${CUDA_ARCHS}) ==="
(cd "${GPUIBS_DIR}" && FASTIBS_CUDA=on CUDA_ARCHS="${CUDA_ARCHS}" bash compile.sh)
"${GPUIBS_DIR}/bin/fastibs" 2>&1 | grep -q "IBS Distance Calculator" \
    && echo "GPU-IBS bin/fastibs OK"
"${GPUIBS_DIR}/bin/fastibs" 2>&1 | grep -q -- "--gpus=N" \
    && echo "GPU-IBS binary has the GPU option block"
echo

# ---- 3. original FastIBS, built natively ----------------------------------------
if [ "${SKIP_ORIG:-0}" = "1" ]; then
    echo "SKIP_ORIG=1: not building the original FastIBS"
    exit 0
fi
echo "=== building the original FastIBS ==="
if [ ! -d "${ORIG_DIR}/.git" ]; then
    git clone "${ORIG_URL}" "${ORIG_DIR}"
fi
cd "${ORIG_DIR}"
echo "upstream commit: $(git rev-parse --short HEAD) ($(git log -1 --format=%cs))"
# Patch only what stops the build outside upstream's Docker mount; the
# sources themselves are untouched, so this is the original algorithm.
sed -i \
    -e 's|set(KMC_LIB_PATH "/project/KMC/bin/libkmc_core.a")|set(KMC_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../KMC/bin/libkmc_core.a")|' \
    -e 's|cmake_minimum_required(VERSION 3.0.0)|cmake_minimum_required(VERSION 3.18)|' \
    src/CMakeLists.txt
grep -q 'CMAKE_CURRENT_SOURCE_DIR}/../KMC/bin/libkmc_core.a' src/CMakeLists.txt \
    || { echo "error: CMake patch did not apply (upstream file changed?)" >&2; exit 1; }
rm -rf build bin
# BOOST_TIMER_ENABLE_DEPRECATED: upstream includes boost/progress.hpp, which
# modern Boost only exposes behind this macro.
cmake -S src -B build -G Ninja \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DCMAKE_CXX_FLAGS="-DBOOST_TIMER_ENABLE_DEPRECATED"
cmake --build build --parallel "$(nproc)"
mkdir -p bin
cp build/fastibs build/fastibsmapper build/KDBIntersect bin/
"${ORIG_DIR}/bin/fastibs" 2>&1 | grep -q "IBS Distance Calculator" && echo "original bin/fastibs OK"
if "${ORIG_DIR}/bin/fastibs" 2>&1 | grep -q -- "--gpus"; then
    echo "error: the 'original' binary shows GPU options -- wrong source tree" >&2
    exit 1
fi
echo
echo "setup complete:"
echo "  GPU-IBS  : ${GPUIBS_DIR}/bin/{fastibs,fastibsmapper,KDBIntersect,probe_bench}"
echo "  original : ${ORIG_DIR}/bin/{fastibs,fastibsmapper,KDBIntersect}"
echo "next: bash scripts/realdata/01_get_data.sh <data_root>"
