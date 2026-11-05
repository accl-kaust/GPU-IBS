#!/bin/bash
#SBATCH --job-name=gpuibs-2gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=run2gpu_%j.out
# Site options (partition, account, GPU type, nodes) go on the sbatch line,
# e.g. sbatch -p <partition> --gres=gpu:a100:2 ...

# Real-data walkthrough, step 4: the two-GPU production run of fastibs.
#
# Two ways to use two devices, both produced by this script:
#   split   (default) two processes, one per GPU (CUDA_VISIBLE_DEVICES), each
#           on half of the chromosomes (<data_root>/reference_gpu0|1 from
#           step 1). Every process writes its own per-chromosome TSVs into
#           the same results folder, so the result set equals a single
#           whole-reference run. This is the mode that actually shortens the
#           wall time, because each device processes half the reference.
#   replica one process with --gpus=2. Since Phase 5 this SPLITS the index
#           across the two devices by prefix range (each holds about half), so
#           it is the mode for a database that does not fit one device; on a
#           single reference it is about as fast as one device (host-bound).
#
# Usage:
#   sbatch [-p <partition>] --gres=gpu:a100:2 scripts/realdata/04_run_2gpu.sbatch <data_root> [RUN] [window]
#   MODE=split|replica|both (default split)   CONDA_ENV=gpuibs   FASTIBS_BIN=bin/fastibs
# Results: <data_root>/FastIBS_runs/<RUN>_w<window>_<mode>/<RUN>_v_MorexV3_<chr>_<window>.tsv
#
# Container equivalents (same binaries inside fastibs.sif, GPU via --nv):
#   replica: bash run.sh <RUN> <data_root> <window> --gpus=2
#   split:   CUDA_VISIBLE_DEVICES=0 apptainer exec --nv --bind $PWD:/project,<data_root>:/project/data fastibs.sif \
#              /project/bin/fastibs /project/data/kmc_sets/<RUN> /project/data/reference_gpu0 /project/data/FastIBS_runs <window> --gpu &
#            CUDA_VISIBLE_DEVICES=1 ... /project/data/reference_gpu1 ... & wait
set -euo pipefail

DATA=${1:?usage: sbatch 04_run_2gpu.sbatch <data_root> [RUN] [window]}
RUN=${2:-ERR6098905}
W=${3:-50000}
MODE=${MODE:-split}
CONDA_ENV=${CONDA_ENV:-gpuibs}

cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../..}}"
DATA=$(cd "$DATA" && pwd)
DB="$DATA/kmc_sets/$RUN"
BIN=${FASTIBS_BIN:-bin/fastibs}
[ -x "$BIN" ] || { echo "error: $BIN not found under $(pwd); run 00_setup.sh" >&2; exit 1; }
[ -d "$DB" ] || { echo "error: $DB missing; run 02_build_kmc.sbatch" >&2; exit 1; }
for d in reference reference_gpu0 reference_gpu1; do
    [ -d "$DATA/$d" ] || { echo "error: $DATA/$d missing; run 01_get_data.sh" >&2; exit 1; }
done

if command -v conda >/dev/null 2>&1; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" 2>/dev/null || true
    set -u
    export LD_LIBRARY_PATH="${CONDA_PREFIX:-/nonexistent}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "=== node: $(hostname), $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
nvidia-smi --query-gpu=index,clocks.applications.graphics,clocks.max.sm,power.limit,clocks_throttle_reasons.active --format=csv || true
NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs visible: ${NGPU}; CPUs in allocation: $(nproc); --mem: ${SLURM_MEM_PER_NODE:-?} MiB"
echo
echo "=== pre-flight: the whole index on one device (what the split mode's two processes need) ==="
WHOLE_OK=1
REQUIRE_FIT=1 bash scripts/realdata/03_preflight.sh "$DB" || WHOLE_OK=0
echo "=== pre-flight: the index split across two devices (--gpus=2) ==="
SHARDS=2 REQUIRE_FIT=1 bash scripts/realdata/03_preflight.sh "$DB" | tail -4 \
    || { echo "error: the index does not fit two devices even when split"; exit 2; }
echo
[ "$NGPU" -ge 2 ] || { echo "error: this run needs two visible GPUs (have $NGPU)"; exit 1; }

run_split () {
    local out="$DATA/FastIBS_runs/${RUN}_w${W}_split"
    rm -rf "$out"; mkdir -p "$out"
    echo "=== split: two processes, one per GPU ==="
    local t0 t1 rc0=0 rc1=0
    t0=$(date +%s.%N)
    CUDA_VISIBLE_DEVICES=0 "$BIN" "$DB" "$DATA/reference_gpu0" "$out" "$W" --gpu > "$out.gpu0.log" 2>&1 &
    local p0=$!
    CUDA_VISIBLE_DEVICES=1 "$BIN" "$DB" "$DATA/reference_gpu1" "$out" "$W" --gpu > "$out.gpu1.log" 2>&1 &
    local p1=$!
    wait "$p0" || rc0=$?
    wait "$p1" || rc1=$?
    t1=$(date +%s.%N)
    for g in 0 1; do
        echo "-- GPU $g: $(grep -m1 'K-mer path' "$out.gpu$g.log" | cut -d: -f2- | xargs || echo '? (no K-mer path line)')"
        grep -E 'index read\+repack|index resident|Reference processed' "$out.gpu$g.log" | sed 's/^/     /' || true
        grep -qF '(GPU-BS)' "$out.gpu$g.log" || { echo "ERROR: GPU $g did not take the GPU-BS path (see $out.gpu$g.log)"; rc0=1; }
    done
    echo "two-GPU split wall time: $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}') s  (exit codes $rc0/$rc1)"
    echo "outputs: $(ls "$out" | wc -l) files in $out"
    [ -s log.txt ] && { echo "fastibs logged errors in $(pwd)/log.txt:"; tail -5 log.txt; }
    return $((rc0 + rc1))
}

run_replica () {
    local out="$DATA/FastIBS_runs/${RUN}_w${W}_replica"
    rm -rf "$out"; mkdir -p "$out"
    echo "=== --gpus=2: one process, the index split across both devices ==="
    local t0 t1 rc=0
    t0=$(date +%s.%N)
    "$BIN" "$DB" "$DATA/reference" "$out" "$W" --gpu --gpus=2 > "$out.log" 2>&1 || rc=$?
    t1=$(date +%s.%N)
    echo "-- $(grep -m1 'K-mer path' "$out.log" | cut -d: -f2- | xargs || echo '? (no K-mer path line)')"
    grep -E 'index read\+repack|index resident|Reference processed' "$out.log" | sed 's/^/     /' || true
    grep -qF '(GPU-BS)' "$out.log" || { echo "ERROR: did not take the GPU-BS path (see $out.log)"; rc=1; }
    echo "--gpus=2 wall time: $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}') s  (exit code $rc)"
    echo "outputs: $(ls "$out" | wc -l) files in $out"
    return $rc
}

RC=0
SPLIT_RAN=0
run_split_if_fits () {
    if [ "$WHOLE_OK" = 1 ]; then SPLIT_RAN=1; run_split; else
        echo "=== split mode skipped: the whole index does not fit one device; the --gpus=2 run (index split) is the two-GPU mode here"
    fi
}
case "$MODE" in
    split)   run_split_if_fits || RC=$? ;;
    replica) run_replica || RC=$? ;;
    both)    run_split_if_fits || RC=$?; echo; run_replica || RC=$? ;;
    *) echo "error: MODE must be split|replica|both" >&2; exit 1 ;;
esac
if [ "$MODE" = both ] && [ "$RC" -eq 0 ] && [ "$SPLIT_RAN" = 1 ]; then
    echo
    if diff -r "$DATA/FastIBS_runs/${RUN}_w${W}_split" "$DATA/FastIBS_runs/${RUN}_w${W}_replica" > /dev/null; then
        echo "split vs replica outputs: byte-identical"
    else
        echo "split vs replica outputs: DIFFER"; RC=1
    fi
fi
echo
exit "$RC"
