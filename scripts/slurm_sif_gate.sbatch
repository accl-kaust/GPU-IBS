#!/bin/bash
#SBATCH --job-name=gpuibs-sif
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=sif_%j.out
# Site options (partition, account, GPU type, nodes) go on the sbatch line,
# e.g. sbatch -p <partition> --gres=gpu:a100:1 ...

# Container gate: exercises a packaged image on a GPU node -- fastibs.sif of
# install.sh (tools in the project's bin/, reached as /project/bin) or the
# self-contained image of the Dockerfile converted to a .sif (tools on PATH
# inside the image; SIF=gpu-ibs.sif selects it):
#   1. the GPU is visible inside the container (--nv) and fastibs took the
#      GPU path there
#   2. the container's GPU output is byte-identical to the container's CPU
#      output (--cpu) for the same reference
#   3. when a native build exists and runs on this host (build/fastibs), its
#      output is identical too
#
# Usage:
#   sbatch [-w node | -x nodes] scripts/slurm_sif_gate.sbatch <kmc_db_folder> <ref_folder> [windowSize]
#   SIF=<image.sif> sbatch ... (default: SIF_NAME of init/config.sh, fastibs.sif)
# Paths may be absolute or relative to the project root; the project root is
# bind-mounted at /project inside the container, so both must lie under it.

set -euo pipefail

DB_DIR=${1:?usage: sbatch slurm_sif_gate.sbatch <kmc_db_folder> <ref_folder> [windowSize]}
REF_DIR=${2:?missing <ref_folder>}
WINDOW=${3:-50000}

cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}}"
source init/config.sh   # SIF_NAME, CONTAINER_CMD
SIF_NAME=${SIF:-${SIF_NAME}}

if [ ! -f "${SIF_NAME}" ]; then
    echo "error: ${SIF_NAME} not found under $(pwd); run install.sh first (or set SIF=)." >&2
    exit 1
fi
if ! command -v "${CONTAINER_CMD}" >/dev/null 2>&1; then
    echo "error: ${CONTAINER_CMD} not found on this node." >&2
    exit 1
fi
# The self-contained image carries fastibs on its PATH; fastibs.sif of
# install.sh runs the project's bin/fastibs through the /project bind.
if "${CONTAINER_CMD}" exec "${SIF_NAME}" sh -c 'command -v fastibs' > /dev/null 2>&1; then
    TOOL=fastibs
else
    TOOL=/project/bin/fastibs
fi

for d in "${DB_DIR}" "${REF_DIR}"; do
    if [ ! -d "${d}" ]; then
        echo "error: folder not found: ${d}" >&2
        exit 1
    fi
done

# Container-side paths: everything is reached through the /project bind.
rel () { realpath --relative-to="$(pwd)" "$1"; }
DB_REL=$(rel "${DB_DIR}")
REF_REL=$(rel "${REF_DIR}")
case "${DB_REL}${REF_REL}" in
    ..*|*/../*) echo "error: database and reference folders must lie under $(pwd)" >&2; exit 1 ;;
esac

echo "=== node: $(hostname), image ${SIF_NAME}, tool ${TOOL} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=== inside ${SIF_NAME} (--nv) ==="
"${CONTAINER_CMD}" exec --nv "${SIF_NAME}" nvidia-smi --query-gpu=name,driver_version --format=csv \
    || echo "(nvidia-smi not usable inside the container)"
echo

WORK="sif_gate_${SLURM_JOB_ID:-local}"
rm -rf "${WORK}"; mkdir -p "${WORK}"
FAIL=0

run_case () { # label outdir extra-args...
    local label=$1 out=$2; shift 2
    local t0 t1
    t0=$(date +%s.%N)
    "${CONTAINER_CMD}" exec --nv --bind "$(pwd)":/project "${SIF_NAME}" \
        "${TOOL}" "/project/${DB_REL}" "/project/${REF_REL}" "/project/${out}" "${WINDOW}" "$@" \
        > "${WORK}/${label}.log" 2>&1
    t1=$(date +%s.%N)
    echo "${label}: total $(echo "${t1} ${t0}" | awk '{printf "%.2f", $1-$2}') s"
    grep -F -e "K-mer path" -e "[gpu]" -e "Reference processed" "${WORK}/${label}.log" || true
}

echo "=== container runs, window=${WINDOW} ==="
run_case gpu "${WORK}/gpu"
if ! grep -qF "(GPU-BS)" "${WORK}/gpu.log"; then
    echo "SIF GATE FAILED: fastibs did not take the GPU path inside the container (see ${WORK}/gpu.log)"
    exit 1
fi
run_case cpu "${WORK}/cpu" --cpu
echo

echo "=== bit-exactness ==="
if diff -r "${WORK}/gpu" "${WORK}/cpu" > /dev/null; then
    echo "container GPU vs container CPU: identical"
else
    echo "container GPU vs container CPU: DIFFER"
    FAIL=1
fi

# Optional: a native build that runs on this host (the container-built
# binaries in build/ may need a newer glibc than the host has; a conda build
# runs fine). Compared against the container's GPU output.
if [ -x build/fastibs ] && { build/fastibs 2>&1 || true; } | grep -q "IBS Distance Calculator"; then
    rm -rf "${WORK}/native"
    if build/fastibs "${DB_DIR}" "${REF_DIR}" "${WORK}/native" "${WINDOW}" > "${WORK}/native.log" 2>&1; then
        grep -F -e "K-mer path" -e "Reference processed" "${WORK}/native.log" || true
        if diff -r "${WORK}/gpu" "${WORK}/native" > /dev/null; then
            echo "container GPU vs native: identical"
        else
            echo "container GPU vs native: DIFFER"
            FAIL=1
        fi
    else
        echo "(native build/fastibs failed to run; see ${WORK}/native.log)"
    fi
else
    echo "(no native build/fastibs runnable on this host; skipping the native comparison)"
fi

echo
if [ "${FAIL}" -ne 0 ]; then
    echo "SIF GATE FAILED (outputs kept at ${WORK})"
    exit 1
fi
echo "SIF GATE PASSED: GPU path ran inside ${SIF_NAME}, outputs identical (kept at ${WORK})"
