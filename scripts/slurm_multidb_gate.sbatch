#!/bin/bash
#SBATCH --job-name=gpuibs-multidb-gate
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=multidb_gate_%j.out
# On-GPU gate of panel mode (docs/MULTIDB_SCAN.md section 9): the device
# path of the multi-database scan -- the multi-database fused kernel on one
# device (--gpus=1) and the routed kernels with every database split by one
# joint cut (--gpus=2) -- must give, for N in {2, 3, 5} databases and
# W in {10000, 50000}, TSVs byte-identical to N single runs, at k=25 and
# k=31, with rounds forced (--resident=2) and across two references. The
# CPU panel path (--cpu) runs the same gate as a control. Builds the panel
# corpora with scripts/make_test_panel.sh when missing (needs kmc).
# Gate job 54696020 ran on Ibex: sbatch --gres=gpu:a100:2 --constraint=a100 ...;
# elsewhere add the site's options (partition, account, GPU type) on the sbatch line.
# CONDA_SH: conda's shell hook (default: that of the conda on PATH, else ~/miniconda3)
#   sbatch scripts/slurm_multidb_gate.sbatch [corpus_root]     (default: ./multidb_corpus)
#   BIN=build/fastibs  CONDA_ENV=gpuibs
set -uo pipefail
ROOT=${1:-multidb_corpus}
cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}}"
CONDA_SH=${CONDA_SH:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh}
if [ -f "$CONDA_SH" ]; then
    set +u; source "$CONDA_SH"
    if conda activate "${CONDA_ENV:-gpuibs}" 2>/dev/null; then
        export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
    else
        echo "(conda env ${CONDA_ENV:-gpuibs} not found; using the current environment)"
    fi
    set -u
fi
BIN=${BIN:-build/fastibs}
[ -x "$BIN" ] || { echo "error: $BIN not found; build first" >&2; exit 1; }
echo "=== node $(hostname), $(date)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)

for spec in "k25 25 1" "k31 31 7"; do
    set -- $spec
    [ -d "$ROOT/$1/db5" ] || bash scripts/make_test_panel.sh "$ROOT/$1" 5 "$2" 2000000 "$3" || exit 1
done

status=0
run () { # label corpus options...
    local label=$1 corpus=$2; shift 2
    echo; echo "##### $label"
    KEEP=1 WORK=$PWD/multidb_gate_${SLURM_JOB_ID:-$$}_$label bash scripts/gate_multidb.sh "$BIN" "$corpus" "$@" || status=1
}
for k in k25 k31; do
    run "${k}_gpu1" "$ROOT/$k" --gpu --gpus=1
    if [ "$ngpu" -ge 2 ]; then run "${k}_gpu2" "$ROOT/$k" --gpu --gpus=2; fi
    run "${k}_cpu" "$ROOT/$k" --cpu
done
# One panel log per device configuration, for the record: which path ran.
for f in multidb_gate_${SLURM_JOB_ID:-$$}_*/panel_5_50000/log; do
    [ -f "$f" ] && { echo "--- $f"; grep -E "^K-mer path|^\[gpu\] pipeline|^\[gpu\] device|^\[panel\]" "$f" | head -12; }
done
echo; [ $status = 0 ] && echo "=== MULTIDB GATE PASSED" || echo "=== MULTIDB GATE FAILED"
exit $status
