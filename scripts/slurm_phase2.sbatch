#!/bin/bash
#SBATCH --job-name=gpuibs-phase2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=phase2_%j.out
# Site options (partition, account, GPU type, nodes) go on the sbatch line,
# e.g. sbatch -p <partition> --gres=gpu:a100:1 ...

# Phase 2 validation gate on a GPU node:
#   1. bit-exactness: streamed GPU pipeline vs CPU (FASTIBS_GPU=0), and vs
#      the Phase 1 per-window sync path (--sync)
#   2. end-to-end timing: CPU-OPT vs GPU-sync vs GPU-streamed, across
#      window sizes — the pipeline's job is to beat GPU-sync wall time
#   3. optional: an Nsight Systems capture of one streamed run when nsys
#      is on PATH (for the H2D/kernel/D2H overlap exit criterion)
#
# Usage:
#   sbatch scripts/slurm_phase2.sbatch <kmc_db_folder> <ref_folder> [windowSize]

set -euo pipefail

DB_DIR=${1:?usage: sbatch slurm_phase2.sbatch <kmc_db_folder> <ref_folder> [windowSize]}
REF_DIR=${2:?missing <ref_folder>}
WINDOW=${3:-50000}
CONDA_ENV=${CONDA_ENV:-gpuibs}

cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}}"
if [ ! -x build/fastibs ]; then
    echo "error: build/fastibs not found under $(pwd); build first." >&2
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
BUILD_DIR=build

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# Device health (see slurm_phase3.sbatch: job 58668's device ran at a third
# of the usual random-access throughput with normal upload times).
# Split into three commands: nvidia-smi rejects a whole --query-gpu if one
# field is unknown to the driver (job 59626: remapped_rows.pending on 535).
nvidia-smi --query-gpu=index,clocks.applications.graphics,clocks.default_applications.graphics,clocks.applications.memory,clocks.max.sm,clocks.max.memory,power.limit,power.max_limit --format=csv || true
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total,clocks_throttle_reasons.active --format=csv || true
nvidia-smi -q -d ROW_REMAPPER 2>/dev/null | grep -E 'Remapped Rows|Correctable Error|Uncorrectable Error|Pending|Remapping Failure' || true
echo
# Device health under load: the ported-index probe rate and the GUPS
# ceiling of this device. A healthy A100-40GB gives ~1.3 Gq/s and
# ~12.5 Gload/s on the large corpus (~3.2 and ~70 on the small one);
# job 58668's device gave a fifth and a third of that.
if [ -x "${BUILD_DIR}/probe_bench" ]; then
    PRE_FILE=$(ls "${DB_DIR}"/*.kmc_pre | head -1)
    echo "=== device health (probe_bench on ${PRE_FILE%.kmc_pre}) ==="
    "${BUILD_DIR}/probe_bench" "${PRE_FILE%.kmc_pre}" 2>&1 | grep -E 'device|^BS |GUPS ceiling' || true
fi
echo

run_case () { # name outdir extra-args...
    local name=$1 out=$2; shift 2
    rm -rf "$out"; mkdir -p "$out"
    local t0=$(date +%s.%N)
    "${BUILD_DIR}/fastibs" "${DB_DIR}" "${REF_DIR}" "$out" "${WINDOW}" "$@" \
        > "$out/run.log" 2>&1
    local t1=$(date +%s.%N)
    local dbload=$(grep -o 'Database loaded in [0-9.]*' "$out/run.log" | grep -o '[0-9.]*' | head -1)
    local proc=$(grep -o 'Reference processed in [0-9.]*' "$out/run.log" | grep -o '[0-9.]*' | head -1)
    local idx=$(grep -o 'index read+repack: [^)]*)' "$out/run.log" | head -1)
    echo "$name: total $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}') s | db-load ${dbload:-?} s | processing ${proc:-?} s${idx:+ | $idx}  ($(grep 'K-mer path' "$out/run.log" | cut -d: -f2- | xargs))"
}

WORK=$(mktemp -d)
echo "=== end-to-end, window=${WINDOW} ==="
run_case "CPU-OPT     " "$WORK/cpu"      --cpu
run_case "GPU-sync    " "$WORK/sync"     --gpu --sync
run_case "GPU-streamed" "$WORK/streamed" --gpu

# The streamed and sync runs must actually have taken their GPU paths.
if ! grep -q "streamed (GPU-BS)" "$WORK/streamed/run.log"; then
    echo "PHASE 2 GATE FAILED: streamed GPU path was not taken"
    exit 1
fi
if ! grep -q "(GPU-BS sync)" "$WORK/sync/run.log"; then
    echo "PHASE 2 GATE FAILED: sync GPU path was not taken"
    exit 1
fi

echo
echo "=== bit-exactness ==="
FAIL=0
for pair in "cpu streamed" "sync streamed"; do
    set -- $pair
    if diff -r -x run.log "$WORK/$1" "$WORK/$2" > /dev/null; then
        echo "$1 vs $2: identical"
    else
        echo "$1 vs $2: DIFFER"
        FAIL=1
    fi
done
if [ "$FAIL" -ne 0 ]; then
    echo "PHASE 2 GATE FAILED: outputs differ (kept at $WORK)"
    exit 1
fi

echo
echo "=== window-size sweep (GPU-streamed vs CPU-OPT) ==="
for W in 10000 100000 1000000; do
    WINDOW=$W
    run_case "CPU-OPT      w=$W" "$WORK/c$W" --cpu
    run_case "GPU-streamed w=$W" "$WORK/g$W" --gpu
    if ! diff -r -x run.log "$WORK/c$W" "$WORK/g$W" > /dev/null; then
        echo "PHASE 2 GATE FAILED: outputs differ at window=$W"
        exit 1
    fi
done

if command -v nsys >/dev/null 2>&1; then
    echo
    echo "=== nsys capture (overlap analysis) ==="
    rm -rf "$WORK/nsys"; mkdir -p "$WORK/nsys"
    nsys profile -o "phase2_overlap_%q{SLURM_JOB_ID}" --force-overwrite=true \
        "${BUILD_DIR}/fastibs" "${DB_DIR}" "${REF_DIR}" "$WORK/nsys" 50000 --gpu \
        > /dev/null 2>&1 || echo "(nsys capture failed; skipping)"
    ls -la phase2_overlap_*.nsys-rep 2>/dev/null || true
else
    echo
    echo "(nsys not on PATH; skipping overlap capture)"
fi

rm -rf "$WORK"
echo
echo "PHASE 2 GATE PASSED: streamed GPU output bit-exact vs CPU and GPU-sync"
