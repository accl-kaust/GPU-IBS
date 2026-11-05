#!/bin/bash
#SBATCH --job-name=gpuibs-phase3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=phase3_%j.out
# Site options (partition, account, GPU type, nodes) go on the sbatch line,
# e.g. sbatch -p <partition> --gres=gpu:a100:1 ...

# Phase 3 ablation gate: fills the paper's probe-structure/ordering matrix
# on one device and checks every configuration stays bit-exact.
#   1. probe_bench matrix: BS, BS+pinned-LUT, BS-SORTED, HASH at three
#      load factors (throughput, footprint, build time)
#   2. end-to-end fastibs: GPU-BS vs GPU-BS-SORTED vs GPU-HASH — wall and
#      processing time, each diffed against the GPU-BS output (itself
#      bit-exact vs CPU per the Phase 2 gate)
#
# Usage:
#   sbatch scripts/slurm_phase3.sbatch <kmc_db_folder> <ref_folder> [windowSize]

set -euo pipefail

DB_DIR=${1:?usage: sbatch slurm_phase3.sbatch <kmc_db_folder> <ref_folder> [windowSize]}
REF_DIR=${2:?missing <ref_folder>}
WINDOW=${3:-50000}
CONDA_ENV=${CONDA_ENV:-gpuibs}

cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}}"
if [ ! -x build/fastibs ] || [ ! -x build/probe_bench ]; then
    echo "error: build/fastibs or build/probe_bench not found under $(pwd); build first." >&2
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
# Device health. Job 58668 landed on a nominally identical A100 that
# sustained a fifth of the probe throughput and a third of the GUPS
# ceiling measured on the 40 GB node, with upload and hash build normal
# and the slowdown growing with a kernel's compute intensity: a clock-
# capped device. Application/max clocks, power limits and memory-health
# state are what distinguish that from a code regression.
# Split into three commands: nvidia-smi rejects a whole --query-gpu if one
# field is unknown to the driver (job 59626: remapped_rows.pending on 535).
nvidia-smi --query-gpu=index,clocks.applications.graphics,clocks.default_applications.graphics,clocks.applications.memory,clocks.max.sm,clocks.max.memory,power.limit,power.max_limit --format=csv || true
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total,clocks_throttle_reasons.active --format=csv || true
nvidia-smi -q -d ROW_REMAPPER 2>/dev/null | grep -E 'Remapped Rows|Correctable Error|Uncorrectable Error|Pending|Remapping Failure' || true
echo

PRE_FILE=$(ls "${DB_DIR}"/*.kmc_pre | head -1)
DB_PREFIX=${PRE_FILE%.kmc_pre}

echo "=== probe_bench ablation matrix on ${DB_PREFIX} ==="
echo "--- BS (baseline) + BS-SORTED ---"
"${BUILD_DIR}/probe_bench" "${DB_PREFIX}" --sorted
echo "(device-health reference: a healthy A100-40GB reached ~12.5 Gload/s at DRAM scale"
echo " and ~70 Gload/s at L2 scale in earlier gates; a far lower ceiling means a throttled"
echo " or shared device and the job's numbers are not comparable)"
echo
echo "--- BS + pinned LUT ---"
"${BUILD_DIR}/probe_bench" "${DB_PREFIX}" --pin-lut | grep -E 'BS|LUT'
echo
for LF in 0.5 0.6 0.8; do
    echo "--- HASH lf=${LF} ---"
    "${BUILD_DIR}/probe_bench" "${DB_PREFIX}" --hash-lf=${LF} | grep -E 'hash backend|HASH'
done
echo

run_case () { # name outdir extra-args...
    local name=$1 out=$2; shift 2
    rm -rf "$out"; mkdir -p "$out"
    local t0=$(date +%s.%N)
    "${BUILD_DIR}/fastibs" "${DB_DIR}" "${REF_DIR}" "$out" "${WINDOW}" "$@" \
        > "$out/run.log" 2>&1
    local t1=$(date +%s.%N)
    local proc=$(grep -o 'Reference processed in [0-9.]*' "$out/run.log" | grep -o '[0-9.]*' | head -1)
    local idx=$(grep -o 'index read+repack: [^)]*)' "$out/run.log" | head -1)
    echo "$name: total $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}') s | processing ${proc:-?} s${idx:+ | $idx}  ($(grep 'K-mer path' "$out/run.log" | cut -d: -f2- | xargs))"
}

WORK=$(mktemp -d)
# Untimed warm-up. The first fastibs run on a node pays the cold read of
# the 2 GB reference (the index is already warm from probe_bench), and it
# used to be the panel's GPU-BS reference: job 58668 measured it at 19.0 s
# against 13.6 s for the next, near-identical configuration. Warm the
# reference, then time every case on equal footing.
mkdir -p "$WORK/warm"
"${BUILD_DIR}/fastibs" "${DB_DIR}" "${REF_DIR}" "$WORK/warm" "${WINDOW}" --gpu --gpus=1 \
    > "$WORK/warm.log" 2>&1 || true
rm -rf "$WORK/warm"
echo "=== end-to-end, window=${WINDOW} (after an untimed warm-up pass) ==="
# One device, whatever the allocation: the ablations are single-device
# (--sort-queries applies only to an unsplit index), and with several
# visible GPUs --gpu alone would split the index and drop the sort.
run_case "GPU-BS       " "$WORK/bs"       --gpu --gpus=1
run_case "GPU-BS pinned" "$WORK/bspin"    --gpu --gpus=1 --pin-lut
run_case "GPU-BS-SORTED" "$WORK/bssorted" --gpu --gpus=1 --sort-queries
run_case "GPU-HASH     " "$WORK/hash"     --gpu --gpus=1 --probe=hash
run_case "GPU-HASH-SORT" "$WORK/hashsort" --gpu --gpus=1 --probe=hash --sort-queries

# Every case must actually have taken its intended GPU path — on any
# device-side failure the binary falls back to the (bit-exact) CPU path,
# which would make the diffs below pass vacuously and mislabel CPU times
# as GPU ablation numbers. grep -F with the parentheses is exact:
# "(GPU-BS)" does not match "(GPU-BS-SORTED)".
check_path () { # outdir expected
    if ! grep -qF "($2)" "$1/run.log"; then
        echo "PHASE 3 GATE FAILED: $(basename "$1") did not take the $2 path"
        exit 1
    fi
}
check_path "$WORK/bs"       "GPU-BS"
check_path "$WORK/bspin"    "GPU-BS"
check_path "$WORK/bssorted" "GPU-BS-SORTED"
check_path "$WORK/hash"     "GPU-HASH"
check_path "$WORK/hashsort" "GPU-HASH-SORTED"

echo
echo "=== bit-exactness (vs GPU-BS) ==="
FAIL=0
for c in bspin bssorted hash hashsort; do
    if diff -r -x run.log "$WORK/bs" "$WORK/$c" > /dev/null; then
        echo "bs vs $c: identical"
    else
        echo "bs vs $c: DIFFER"
        FAIL=1
    fi
done
if [ "$FAIL" -ne 0 ]; then
    echo "PHASE 3 GATE FAILED: outputs differ (kept at $WORK)"
    exit 1
fi

rm -rf "$WORK"
echo
echo "PHASE 3 GATE PASSED: all probe configurations bit-exact"
