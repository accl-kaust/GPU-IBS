#!/bin/bash
#SBATCH --job-name=gpuibs-phase4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=phase4_%j.out
# Site options (partition, account, GPU type, nodes) go on the sbatch line,
# e.g. sbatch -p <partition> --gres=gpu:a100:4 ...

# Phase 4/5 validation gate on a multi-GPU node:
#   1. mapper: GPU coverage output byte-identical to the CPU path, on one
#      device and with the index split across all of them
#   2. intersection: GPU count == CPU count (cross) and == total (self);
#      with several GPUs the larger database is split across them
#   3. multi-GPU: fastibs output identical across --gpus=1..N (device d
#      holding shard d of the index), with the strong-scaling timing table
#
# Usage:
#   sbatch scripts/slurm_phase4.sbatch <kmc_db_folder> <ref_folder> [windowSize] [second_db_folder]

set -euo pipefail

DB_DIR=${1:?usage: sbatch slurm_phase4.sbatch <kmc_db_folder> <ref_folder> [windowSize] [second_db_folder]}
REF_DIR=${2:?missing <ref_folder>}
WINDOW=${3:-50000}
DB2_DIR=${4:-}
CONDA_ENV=${CONDA_ENV:-gpuibs}

cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}}"
if [ ! -x build/fastibs ] || [ ! -x build/fastibsmapper ] || [ ! -x build/KDBIntersect ]; then
    echo "error: build/ binaries not found under $(pwd); build first." >&2
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
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
# Device health (see slurm_phase3.sbatch: job 58668's device ran at a third
# of the usual random-access throughput with normal upload times).
# Split into three commands: nvidia-smi rejects a whole --query-gpu if one
# field is unknown to the driver (job 59626: remapped_rows.pending on 535).
nvidia-smi --query-gpu=index,clocks.applications.graphics,clocks.default_applications.graphics,clocks.applications.memory,clocks.max.sm,clocks.max.memory,power.limit,power.max_limit --format=csv || true
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total,clocks_throttle_reasons.active --format=csv || true
nvidia-smi -q -d ROW_REMAPPER 2>/dev/null | grep -E 'Remapped Rows|Correctable Error|Uncorrectable Error|Pending|Remapping Failure' || true
NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs visible: ${NGPU}"
echo
# Device health under load: the ported-index probe rate and the GUPS
# ceiling of each device. A healthy A100-40GB gives ~1.3 Gq/s and
# ~12.5 Gload/s on the large corpus (~3.2 and ~70 on the small one);
# job 58668's device gave a fifth and a third of that.
if [ -x "${BUILD_DIR}/probe_bench" ]; then
    PRE_FILE=$(ls "${DB_DIR}"/*.kmc_pre | head -1)
    echo "=== device health (probe_bench on ${PRE_FILE%.kmc_pre}, each GPU) ==="
    for G in $(seq 0 $((NGPU-1))); do
        echo "-- GPU $G"
        CUDA_VISIBLE_DEVICES=$G "${BUILD_DIR}/probe_bench" "${PRE_FILE%.kmc_pre}" 2>&1 \
            | grep -E 'device|^BS |GUPS ceiling' || true
    done
    echo
fi

WORK=$(mktemp -d)
FAIL=0

# --- 1. mapper --------------------------------------------------------------
echo "=== mapper: GPU vs CPU ==="
# Both paths read the same ~12 GB index and reference, so whichever runs
# first pays the cold page cache and the other does not -- a bias worth
# more than the effect being measured. Each path therefore gets an
# untimed warm-up pass, and only the second (warm) pass is reported.
run_mapper () { # label outdir env-prefix [fastibsmapper options...]
    local label=$1 out=$2 envp=$3
    shift 3
    rm -rf "$out" "$out.warm"; mkdir -p "$out" "$out.warm"
    env ${envp} "${BUILD_DIR}/fastibsmapper" "${DB_DIR}" "${REF_DIR}" "$out.warm" "$@" \
        > "$WORK/$label.warm.log" 2>&1
    local t0=$(date +%s.%N)
    env ${envp} "${BUILD_DIR}/fastibsmapper" "${DB_DIR}" "${REF_DIR}" "$out" "$@" \
        > "$WORK/$label.log" 2>&1
    local t1=$(date +%s.%N)
    echo "$label mapper: $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}') s (warm)  ($(grep 'K-mer path' "$WORK/$label.log" | cut -d: -f2- | xargs))"
    grep -H 'Mapping phases' "$WORK/$label.log" | sed "s|$WORK/||" || true
    # Both passes' index-read lines: a timed pass that reads far slower
    # than its own warm-up pass means the page cache did not hold between
    # them (job 58561: 46 s at 260 MB/s where the scan job saw 2 s).
    grep -H 'index read' "$WORK/$label.warm.log" "$WORK/$label.log" | sed "s|$WORK/||" || true
    grep -H 'Database loaded' "$WORK/$label.log" | sed "s|$WORK/||" || true
}
run_mapper GPU "$WORK/map_gpu" "" --gpus=1
if ! grep -q "GPU-" "$WORK/GPU.log"; then
    echo "PHASE 4 GATE FAILED: mapper did not take a GPU path"
    exit 1
fi
run_mapper CPU "$WORK/map_cpu" "FASTIBS_GPU=0"
if diff -r "$WORK/map_gpu" "$WORK/map_cpu" > /dev/null; then
    echo "mapper output: byte-identical"
else
    echo "mapper output: DIFFERS"
    FAIL=1
fi
if [ "${NGPU}" -gt 1 ]; then
    # The default uses every visible device, the index split across them.
    run_mapper MULTI "$WORK/map_multi" "" --gpus="${NGPU}"
    if ! grep -q "index split across ${NGPU} devices" "$WORK/MULTI.log"; then
        echo "PHASE 4 GATE FAILED: mapper --gpus=${NGPU} did not split the index"
        FAIL=1
    fi
    if diff -r "$WORK/map_gpu" "$WORK/map_multi" > /dev/null; then
        echo "mapper --gpus=${NGPU} (index split): byte-identical"
    else
        echo "mapper --gpus=${NGPU} (index split): DIFFERS"
        FAIL=1
    fi
fi
echo

# --- 2. intersection --------------------------------------------------------
# A failing KDBIntersect (e.g. mismatched k between the databases) must
# fail the gate loudly, not kill the script inside a $() under set -e.
run_intersect () { # var-name env-prefix db1 db2
    local -n out=$1
    local envp=$2 db1=$3 db2=$4
    if ! out=$(env ${envp} "${BUILD_DIR}/KDBIntersect" "$db1" "$db2" 2> "$WORK/intersect.err" | tail -1) \
       || ! [[ "$out" =~ ^[0-9]+$ ]]; then
        echo "KDBIntersect failed on ($db1, $db2): $(cat "$WORK/intersect.err")"
        out="FAILED"
        FAIL=1
    fi
    # A GPU run must really have probed on the device: KDBIntersect's
    # stdout is the count alone, and on a device error it prints
    # "falling back to CPU intersection" to stderr and then produces the
    # (equal) CPU count -- which would pass the count comparison vacuously.
    # With several GPUs the stderr carries one "index shard i/N resident"
    # line per device instead of the single "index resident" line.
    if [ -z "$envp" ]; then
        if grep -q 'falling back' "$WORK/intersect.err" || ! grep -q 'resident' "$WORK/intersect.err"; then
            echo "PHASE 4 GATE FAILED: GPU intersection ($db1, $db2) did not take the GPU path: $(cat "$WORK/intersect.err")"
            FAIL=1
        fi
        if [ "${NGPU}" -gt 1 ] && [ "$(grep -c 'shard .* resident' "$WORK/intersect.err")" -ne "${NGPU}" ]; then
            echo "PHASE 4 GATE FAILED: GPU intersection ($db1, $db2) did not split the index across ${NGPU} devices: $(cat "$WORK/intersect.err")"
            FAIL=1
        fi
    fi
}
echo "=== intersection: GPU vs CPU ==="
run_intersect SELF_GPU "" "${DB_DIR}" "${DB_DIR}"
run_intersect SELF_CPU "FASTIBS_GPU=0" "${DB_DIR}" "${DB_DIR}"
echo "self: gpu=${SELF_GPU} cpu=${SELF_CPU}"
[ "$SELF_GPU" = "$SELF_CPU" ] || { echo "self-intersection DIFFERS"; FAIL=1; }
if [ -n "$DB2_DIR" ]; then
    run_intersect X_GPU "" "${DB_DIR}" "${DB2_DIR}"
    run_intersect X_CPU "FASTIBS_GPU=0" "${DB_DIR}" "${DB2_DIR}"
    echo "cross: gpu=${X_GPU} cpu=${X_CPU}"
    [ "$X_GPU" = "$X_CPU" ] || { echo "cross-intersection DIFFERS"; FAIL=1; }
fi
echo

# --- 3. multi-GPU strong scaling ---------------------------------------------
# --gpus=G splits the index G ways (device d holds prefix-range shard d and
# searches only its k-mers); every G must reproduce the --gpus=1 output.
echo "=== fastibs strong scaling (index split G ways), window=${WINDOW} ==="
REF_OUT=""
for G in $(seq 1 "${NGPU}"); do
    out="$WORK/g$G"
    rm -rf "$out"; mkdir -p "$out"
    t0=$(date +%s.%N)
    "${BUILD_DIR}/fastibs" "${DB_DIR}" "${REF_DIR}" "$out" "${WINDOW}" --gpu --gpus="$G" \
        > "$out/run.log" 2>&1
    t1=$(date +%s.%N)
    # `|| true`: under set -e a top-level assignment takes its command
    # substitution's status, and a no-match grep would abort the script
    # before the loud path check below gets to run.
    proc=$(grep -o 'Reference processed in [0-9.]*' "$out/run.log" | grep -o '[0-9.]*' | head -1 || true)
    idx=$(grep -o 'index read+repack: [^)]*)' "$out/run.log" | head -1 || true)
    shard=$(grep -o 'uploaded in [0-9.]* ms' "$out/run.log" | head -1 || true)
    echo "--gpus=$G: total $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}') s | processing ${proc:-?} s${idx:+ | $idx}${shard:+ | shard 1 $shard}"
    if ! grep -qF "(GPU-BS)" "$out/run.log"; then
        echo "PHASE 4 GATE FAILED: --gpus=$G did not take the GPU-BS path"
        exit 1
    fi
    if [ "$G" -gt 1 ] && ! grep -q "index split across $G devices" "$out/run.log"; then
        echo "PHASE 4 GATE FAILED: --gpus=$G did not split the index"
        FAIL=1
    fi
    if [ -z "$REF_OUT" ]; then
        REF_OUT="$out"
    elif ! diff -r -x run.log "$REF_OUT" "$out" > /dev/null; then
        echo "PHASE 4 GATE FAILED: --gpus=$G output differs from --gpus=1"
        FAIL=1
    fi
done
echo

if [ "$FAIL" -ne 0 ]; then
    echo "PHASE 4 GATE FAILED (outputs kept at $WORK)"
    exit 1
fi
rm -rf "$WORK"
echo "PHASE 4 GATE PASSED: mapper byte-identical, intersection counts equal, multi-GPU (index split) outputs identical"
