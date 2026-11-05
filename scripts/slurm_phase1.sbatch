#!/bin/bash
#SBATCH --job-name=gpuibs-phase1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=phase1_%j.out
# Site options (partition, account, GPU type, nodes) go on the sbatch line,
# e.g. sbatch -p <partition> --gres=gpu:a100:1 ...

# Phase 1 validation gate on a GPU node:
#   1. probe_bench: sustained probe throughput vs a GUPS-style ceiling
#   2. end-to-end fastibs, GPU vs CPU (FASTIBS_GPU=0) -> bit-exact TSV diff
#
# Usage:
#   sbatch scripts/slurm_phase1.sbatch <kmc_db_folder> <ref_folder> [windowSize]
#
# Generate a test corpus first if you have no production data:
#   bash scripts/make_test_corpus.sh $PWD/corpus 25
#   sbatch scripts/slurm_phase1.sbatch $PWD/corpus/db $PWD/corpus/refs 50000
#
# Assumes the repo was built with -DFASTIBS_CUDA=ON into ./build and that
# the conda env named in CONDA_ENV below has the CUDA runtime installed.

set -euo pipefail

DB_DIR=${1:?usage: sbatch slurm_phase1.sbatch <kmc_db_folder> <ref_folder> [windowSize]}
REF_DIR=${2:?missing <ref_folder>}
WINDOW=${3:-50000}
CONDA_ENV=${CONDA_ENV:-gpuibs}

# Under sbatch the script runs from a spool copy (/var/lib/slurm/...), so
# BASH_SOURCE cannot locate the repo; SLURM_SUBMIT_DIR is the directory
# sbatch was invoked from. Submit from the repo root (or set GPUIBS_DIR).
cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}}"
if [ ! -x build/fastibs ] || [ ! -x build/probe_bench ]; then
    echo "error: build/fastibs or build/probe_bench not found under $(pwd)." >&2
    echo "Submit this job from the repo root (after building), or set GPUIBS_DIR=/path/to/GPU-IBS." >&2
    exit 1
fi

# --- environment ------------------------------------------------------------
# Clusters without an environment-modules system are fine; the CUDA runtime
# comes from the conda env in that case.
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
    # conda's compiler-package activation hooks reference variables that
    # are unset on a fresh shell (CONDA_BACKUP_CXX and friends), which
    # aborts under `set -u`; relax it for the activation only.
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
    # The binaries link conda's libcudart/libstdc++; make sure they are
    # findable even if the build rpath does not cover them.
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Set after conda activation: conda's compiler package exports a variable
# literally named BUILD (the autotools triplet), so a generic name set
# earlier would be clobbered.
BUILD_DIR=build

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo

# --- 1. microbenchmark ------------------------------------------------------
# probe_bench takes the database path *prefix* (no .kmc_pre/.kmc_suf suffix)
PRE_FILE=$(ls "${DB_DIR}"/*.kmc_pre | head -1)
DB_PREFIX=${PRE_FILE%.kmc_pre}
echo "=== probe_bench on ${DB_PREFIX} ==="
"${BUILD_DIR}/probe_bench" "${DB_PREFIX}"
echo

# --- 2. end-to-end bit-exactness gate ---------------------------------------
OUT_GPU=$(mktemp -d)
OUT_CPU=$(mktemp -d)
GPU_LOG=$(mktemp)
echo "=== fastibs GPU run ==="
"${BUILD_DIR}/fastibs" "${DB_DIR}" "${REF_DIR}" "${OUT_GPU}" "${WINDOW}" | tee "${GPU_LOG}"
# Without this check the gate could pass vacuously: when no usable device
# is present the binary falls back to the CPU path, and the diff below
# would then compare CPU output against CPU output.
if ! grep -q "GPU-BS" "${GPU_LOG}"; then
    echo "PHASE 1 GATE FAILED: the GPU path was not taken (fell back to CPU)."
    echo "Check the driver/toolkit versions reported by nvidia-smi above."
    exit 1
fi
rm -f "${GPU_LOG}"
echo
echo "=== fastibs CPU run (FASTIBS_GPU=0) ==="
FASTIBS_GPU=0 "${BUILD_DIR}/fastibs" "${DB_DIR}" "${REF_DIR}" "${OUT_CPU}" "${WINDOW}"
echo

echo "=== diff ==="
if diff -r "${OUT_GPU}" "${OUT_CPU}"; then
    echo "PHASE 1 GATE PASSED: GPU output bit-exact vs CPU"
    STATUS=0
else
    echo "PHASE 1 GATE FAILED: GPU output differs from CPU"
    echo "GPU output kept at ${OUT_GPU}, CPU output at ${OUT_CPU}"
    exit 1
fi
rm -rf "${OUT_GPU}" "${OUT_CPU}"
exit ${STATUS}
