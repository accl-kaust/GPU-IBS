#!/bin/bash
#SBATCH --job-name=kmc-build
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=kmc_%j.out
# Site options (partition, account, nodes) go on the sbatch line, e.g. sbatch -p <partition> ...

# Real-data walkthrough, step 2: count the k-mers of the reads with KMC 3
# into the FastIBS layout <data_root>/kmc_sets/<RUN>/<RUN>.res.kmc_{pre,suf},
# then run the device-footprint pre-flight (step 3) on the result.
#
# Usage:
#   sbatch [--partition=<cpu partition>] scripts/realdata/02_build_kmc.sbatch <data_root> [RUN] [k] [ci]
# Defaults: RUN=ERR6098905, k=25 (the paper's corpora; IBSpy/FastIBS
# production databases often use 31 -- see the walkthrough for the footprint
# consequence), ci=2 (KMC's default: k-mers seen once are dropped).
# Environment:
#   KMC_TMP     scratch for KMC's bins; needs roughly 2-3x the compressed input
#               (default <data_root>/kmc_tmp; point it at a node-local NVMe
#               only if that disk really has ~300 GB free -- a small /tmp
#               fills up and kills the run)
#   KMC_MEM_GB  RAM cap for kmc -m (default: 75% of the job's memory, at most 1024)
#   KMC_RAM_ONLY=1  kmc -r: keep the bins in RAM instead of on disk. Needs a
#               node with RAM for the whole intermediate (order of the
#               compressed input size, 100-250 GB for 200 Gbases; a 1.5 TB
#               node with --mem=1200G is comfortable). Uses no scratch at all.
#   CONDA_ENV   conda env with kmc and kmc_tools (default gpuibs)
#   FORCE=1     rebuild an existing database
set -euo pipefail

DATA=${1:?usage: sbatch 02_build_kmc.sbatch <data_root> [RUN] [k] [ci]}
RUN=${2:-ERR6098905}
K=${3:-25}
CI=${4:-2}
CONDA_ENV=${CONDA_ENV:-gpuibs}

cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../..}}"
DATA=$(cd "$DATA" && pwd)
LIST="$DATA/reads/${RUN}.files"
[ -s "$LIST" ] || { echo "error: $LIST missing -- run 01_get_data.sh first" >&2; exit 1; }

if command -v conda >/dev/null 2>&1; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" 2>/dev/null || true
    set -u
fi
KMC=$(command -v kmc || true)
[ -n "$KMC" ] || { echo "error: kmc not found on PATH (conda env ${CONDA_ENV} from 00_setup.sh, or conda install -c bioconda kmc)" >&2; exit 1; }

CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
if [ -n "${KMC_MEM_GB:-}" ]; then
    MEM_GB=$KMC_MEM_GB
elif [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    MEM_GB=$((SLURM_MEM_PER_NODE * 3 / 4 / 1024))
else
    MEM_GB=100
fi
[ "$MEM_GB" -ge 4 ] || MEM_GB=4
[ "$MEM_GB" -le 1024 ] || MEM_GB=1024      # kmc's -m upper bound
RAM_FLAG=""
[ "${KMC_RAM_ONLY:-0}" = "1" ] && RAM_FLAG="-r"
TMP_ROOT=${KMC_TMP:-$DATA/kmc_tmp}
KTMP="$TMP_ROOT/kmc_${RUN}_${SLURM_JOB_ID:-$$}"
OUT="$DATA/kmc_sets/$RUN"
PREFIX="$OUT/$RUN.res"

echo "=== node: $(hostname), $(date) ==="
echo "kmc: $KMC ($($KMC 2>&1 | grep -m1 -i 'ver' || true))"
echo "input list: $LIST ($(wc -l < "$LIST") files, $(awk '{s+=$1} END{printf "%.1f", s/1e9}' < <(xargs -a "$LIST" stat -c %s)) GB)"
echo "k=$K ci=$CI cs=255 threads=$CPUS mem=${MEM_GB}G tmp=$KTMP ${RAM_FLAG:+(RAM-only mode: bins stay in memory, tmp unused)}"
echo "output: $PREFIX.kmc_{pre,suf}"
if [ -f "$PREFIX.kmc_pre" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "database exists; FORCE=1 to rebuild. Running the pre-flight only."
else
    mkdir -p "$OUT" "$KTMP"
    df -h "$KTMP" | tail -1
    t0=$(date +%s)
    # shellcheck disable=SC2086
    "$KMC" -k"$K" -ci"$CI" -cs255 -m"$MEM_GB" -t"$CPUS" -fq -hp $RAM_FLAG -j"$OUT/kmc_stats.json" \
        @"$LIST" "$PREFIX" "$KTMP" 2>&1 | tee "$OUT/kmc.log"
    t1=$(date +%s)
    rm -rf "$KTMP"
    echo "kmc finished in $((t1 - t0)) s"
fi
echo
echo "=== kmc_tools info ==="
kmc_tools info "$PREFIX" | tee "$OUT/kmc_info.txt"
ls -la "$OUT"
echo
echo "=== device-footprint pre-flight ==="
bash scripts/realdata/03_preflight.sh "$OUT" || true
echo
echo "next: sbatch [-p <partition>] scripts/realdata/04_run_2gpu.sbatch $DATA $RUN 50000"
