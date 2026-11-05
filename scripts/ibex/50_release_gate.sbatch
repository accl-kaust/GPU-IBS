#!/bin/bash
#SBATCH --job-name=gpuibs-release-gate
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=release_gate_%j.out
# Ibex (submit from glogin): the release gate. Builds the checkout with the
# GPU path, then requires every GPU configuration of the three tools to give
# output byte-identical to the CPU path, on two A100s:
#   scan     k=25 and k=31, W=10000 and 50000: --gpus=1 (streamed), --sync,
#            --sort-queries, --pin-lut, --probe=hash, --gpus=2 (peer routing),
#            --gpus=2 --no-peer (host-staged routing), --gpus=2 --probe=hash
#   mapper   --gpus=1, --gpus=2, --gpus=2 --no-peer
#   intersect self and cross pairs: GPU 1/2 devices, staged, hash, with 1 and
#            16 enumeration threads, against the CPU path and FASTIBS_LEGACY=1
#   panel    scripts/gate_multidb.sh at k=25 and k=31: --gpus=1, --gpus=2,
#            --gpus=2 --no-peer, --cpu
#   k > 32   a k=35 database on --gpu must give the original string path's output
#   barley   optional (BARLEY_DB=<KMC folder> BARLEY_REFS=<fasta folder>): one
#            real p=9 database, CPU vs one GPU vs two shards vs staged
# Every GPU run's log must show the GPU path (and the routing it was asked
# for); a run that silently fell back to the CPU counts as a failure.
#   sbatch scripts/ibex/50_release_gate.sbatch            (from the checkout)
#   GATE_DIR=<scratch for corpora and outputs>  (default $IBEX_USER_DIR/gpuibs_data/release_gate)
set -uo pipefail
cd "${GPUIBS_DIR:-${SLURM_SUBMIT_DIR:-.}}"   # GPUIBS_DIR: the checkout, when not submitted from it
# CONDA_SH: conda's shell hook (default: the conda on PATH, else ~/miniconda3, which 00_setup_ibex.sh links to the scratch Miniforge)
CONDA_SH=${CONDA_SH:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh}
set +u; source "$CONDA_SH"; conda activate "${CONDA_ENV:-gpuibs}"; set -u
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# IBEX_USER_DIR: per-user Ibex scratch
IBEX_USER_DIR=${IBEX_USER_DIR:-/ibex/user/$USER}
G=${GATE_DIR:-$IBEX_USER_DIR/gpuibs_data/release_gate}
mkdir -p "$G"
echo "=== node $(hostname), $(date), job ${SLURM_JOB_ID:-}, commit $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m | head -4

echo; echo "##### build"
FASTIBS_CUDA=on bash compile.sh > "$G/build_${SLURM_JOB_ID:-$$}.log" 2>&1 || { tail -40 "$G/build_${SLURM_JOB_ID:-$$}.log"; echo "=== RELEASE GATE FAILED (build)"; exit 1; }
for b in fastibs fastibsmapper KDBIntersect; do [ -x bin/$b ] || { echo "bin/$b missing"; exit 1; }; done
export FASTIBS_THREADS=${FASTIBS_THREADS:-16}
echo "fastibs pool: FASTIBS_THREADS=$FASTIBS_THREADS"

echo; echo "##### corpora"
[ -f "$G/k25/db/sample.kmc_pre" ]  || bash scripts/make_test_corpus.sh "$G/k25" 25 > "$G/mk_k25.log" 2>&1 || { tail "$G/mk_k25.log"; exit 1; }
[ -f "$G/k25b/db/sample.kmc_pre" ] || bash scripts/make_test_corpus.sh "$G/k25b" 25 2000000 2 > "$G/mk_k25b.log" 2>&1 || { tail "$G/mk_k25b.log"; exit 1; }
[ -f "$G/k31/db/sample.kmc_pre" ]  || bash scripts/make_test_corpus.sh "$G/k31" 31 > "$G/mk_k31.log" 2>&1 || { tail "$G/mk_k31.log"; exit 1; }
if [ ! -f "$G/k35/db/sample.kmc_pre" ]; then
    mkdir -p "$G/k35/db" "$G/k35/tmp"
    kmc -k35 -ci1 -fm -t8 "$G/k25/sample_reads.fasta" "$G/k35/db/sample" "$G/k35/tmp" > "$G/mk_k35.log" 2>&1 || { tail "$G/mk_k35.log"; exit 1; }
fi
for c in k25 k25b k31 k35; do ls -l "$G/$c/db/sample.kmc_pre" | awk '{print "    "$NF, $5" bytes"}'; done

status=0; checks=0
ok ()   { checks=$((checks+1)); echo "PASS $*"; }
bad ()  { checks=$((checks+1)); status=1; echo "FAIL $*"; }
# gpu_used <log> [staged|peer|split]: the run took the GPU path, and the routing
# asked for (peer or staged) or, for the intersection, the index split across
# two devices; the intersection routes each query batch to its owner on the
# host, so it prints no routing line
gpu_used () {
    grep -q '^\[gpu\] no CUDA device\|falling back to CPU path' "$1" && return 1
    grep -q '^\[gpu\]' "$1" || return 1
    case "${2:-}" in
        "")    return 0 ;;
        split) grep -q '^\[gpu\] splitting the index across 2 ' "$1" ;;
        *)     grep -q "^\[gpu\] routing: $2" "$1" ;;
    esac
}
O=$G/out_${SLURM_JOB_ID:-$$}; rm -rf "$O"; mkdir -p "$O"

echo; echo "##### scan"
SCAN_CFGS=("g1|--gpu --gpus=1|" "sync|--gpu --gpus=1 --sync|" "sorted|--gpu --gpus=1 --sort-queries|" "pin|--gpu --gpus=1 --pin-lut|"
           "hash1|--gpu --gpus=1 --probe=hash|" "g2|--gpu --gpus=2|peer" "g2staged|--gpu --gpus=2 --no-peer|staged" "hash2|--gpu --gpus=2 --probe=hash|")
for c in k25 k31; do
    for W in 10000 50000; do
        ref=$O/scan_${c}_${W}_cpu; mkdir -p "$ref"
        bin/fastibs "$G/$c/db" "$G/$c/refs" "$ref" $W --cpu > "$ref.log" 2>&1 || bad "scan $c W=$W cpu exited $?"
        for spec in "${SCAN_CFGS[@]}"; do
            IFS='|' read -r name opts route <<< "$spec"
            d=$O/scan_${c}_${W}_$name; mkdir -p "$d"
            bin/fastibs "$G/$c/db" "$G/$c/refs" "$d" $W $opts > "$d.log" 2>&1; rc=$?
            if [ $rc -ne 0 ]; then bad "scan $c W=$W $name exited $rc"; tail -3 "$d.log"
            elif ! gpu_used "$d.log" "$route"; then bad "scan $c W=$W $name did not take the GPU path${route:+ with $route routing}"
            elif diff -r "$ref" "$d" > /dev/null; then ok "scan $c W=$W $name"
            else bad "scan $c W=$W $name output differs"; fi
        done
    done
done

echo; echo "##### mapper"
for c in k25 k31; do
    ref=$O/map_${c}_cpu; mkdir -p "$ref"
    bin/fastibsmapper "$G/$c/db" "$G/$c/refs" "$ref" --cpu > "$ref.log" 2>&1 || bad "mapper $c cpu exited $?"
    for spec in "g1|--gpu --gpus=1|" "g2|--gpu --gpus=2|peer" "g2staged|--gpu --gpus=2 --no-peer|staged"; do
        IFS='|' read -r name opts route <<< "$spec"
        d=$O/map_${c}_$name; mkdir -p "$d"
        bin/fastibsmapper "$G/$c/db" "$G/$c/refs" "$d" $opts > "$d.log" 2>&1; rc=$?
        if [ $rc -ne 0 ]; then bad "mapper $c $name exited $rc"; tail -3 "$d.log"
        elif ! gpu_used "$d.log" "$route"; then bad "mapper $c $name did not take the GPU path${route:+ with $route routing}"
        elif diff -r "$ref" "$d" > /dev/null; then ok "mapper $c $name"
        else bad "mapper $c $name output differs"; fi
    done
done

echo; echo "##### intersect"
for pair in "k25 k25" "k25 k25b" "k25b k25" "k31 k31"; do
    set -- $pair; a=$G/$1/db; b=$G/$2/db; lbl="$1x$2"
    want=$(FASTIBS_LEGACY=1 FASTIBS_GPU=0 bin/KDBIntersect "$a" "$b" 2> "$O/int_${lbl}_legacy.err" | tail -1)
    cpu=$(FASTIBS_GPU=0 bin/KDBIntersect "$a" "$b" 2> "$O/int_${lbl}_cpu.err" | tail -1)
    [ -n "$want" ] && [ "$cpu" = "$want" ] && ok "intersect $lbl cpu = legacy ($want)" || bad "intersect $lbl cpu '$cpu' vs legacy '$want'"
    for spec in "g1|FASTIBS_GPUS=1|" "g2|FASTIBS_GPUS=2|split" "g2staged|FASTIBS_GPUS=2 FASTIBS_NO_PEER=1|split" "hash1|FASTIBS_GPUS=1 FASTIBS_PROBE=hash|" "hash2|FASTIBS_GPUS=2 FASTIBS_PROBE=hash|"; do
        IFS='|' read -r name envs route <<< "$spec"
        for t in 1 16; do
            log=$O/int_${lbl}_${name}_t$t.log
            got=$(env $envs FASTIBS_THREADS=$t bin/KDBIntersect "$a" "$b" 2> "$log.err" | tee "$log" | tail -1)
            cat "$log.err" >> "$log"
            if ! gpu_used "$log" "$route"; then bad "intersect $lbl $name t=$t did not take the GPU path${route:+ ($route)}"
            elif [ "$got" = "$want" ]; then ok "intersect $lbl $name t=$t"
            else bad "intersect $lbl $name t=$t count '$got' vs '$want'"; fi
        done
    done
done

echo; echo "##### k > 32"
d=$O/k35_gpu; l=$O/k35_legacy; mkdir -p "$d" "$l"
bin/fastibs "$G/k35/db" "$G/k25/refs" "$d" 50000 --gpu > "$d.log" 2>&1; rc=$?
FASTIBS_LEGACY=1 bin/fastibs "$G/k35/db" "$G/k25/refs" "$l" 50000 --cpu > "$l.log" 2>&1
grep -h 'k=35' "$d.log" | head -1
[ $rc -eq 0 ] && diff -r "$d" "$l" > /dev/null && ok "k=35 --gpu = original string path" || bad "k=35 --gpu (rc $rc) differs from the string path"
m=$O/k35_map; ml=$O/k35_map_legacy; mkdir -p "$m" "$ml"
bin/fastibsmapper "$G/k35/db" "$G/k25/refs" "$m" --gpu > "$m.log" 2>&1; rc=$?
FASTIBS_LEGACY=1 bin/fastibsmapper "$G/k35/db" "$G/k25/refs" "$ml" --cpu > "$ml.log" 2>&1
[ $rc -eq 0 ] && diff -r "$m" "$ml" > /dev/null && ok "k=35 mapper --gpu = original string path" || bad "k=35 mapper (rc $rc) differs"

echo; echo "##### panel"
for spec in "k25 25 1" "k31 31 7"; do
    set -- $spec
    P=$G/panel_$1
    [ -d "$P/db5" ] || bash scripts/make_test_panel.sh "$P" 5 "$2" 2000000 "$3" > "$G/mk_panel_$1.log" 2>&1 || { tail "$G/mk_panel_$1.log"; bad "panel corpus $1"; continue; }
    for opts in "--gpu --gpus=1" "--gpu --gpus=2" "--gpu --gpus=2 --no-peer" "--cpu"; do
        tag=$(echo "$1 $opts" | tr ' =' '__')
        if KEEP=1 WORK=$O/panel_$tag bash scripts/gate_multidb.sh bin/fastibs "$P" $opts > "$O/panel_$tag.log" 2>&1; then
            if [ "$opts" != "--cpu" ] && ! gpu_used "$O/panel_$tag/panel_5_50000/log" "$( [[ $opts == *no-peer* ]] && echo staged )"; then
                bad "panel $1 $opts did not take the GPU path"
            else ok "panel $1 $opts ($(grep -o '[0-9]* TSV comparisons identical' "$O/panel_$tag.log"))"; fi
        else bad "panel $1 $opts"; tail -5 "$O/panel_$tag.log"; fi
    done
done

if [ -n "${BARLEY_DB:-}" ] && [ -n "${BARLEY_REFS:-}" ]; then
    echo; echo "##### barley ($BARLEY_DB)"
    ref=$O/barley_cpu; mkdir -p "$ref"
    bin/fastibs "$BARLEY_DB" "$BARLEY_REFS" "$ref" 50000 --cpu > "$ref.log" 2>&1 || bad "barley cpu exited $?"
    for spec in "g1|--gpu --gpus=1|" "g2|--gpu --gpus=2|peer" "g2staged|--gpu --gpus=2 --no-peer|staged"; do
        IFS='|' read -r name opts route <<< "$spec"
        d=$O/barley_$name; mkdir -p "$d"
        bin/fastibs "$BARLEY_DB" "$BARLEY_REFS" "$d" 50000 $opts > "$d.log" 2>&1; rc=$?
        if [ $rc -ne 0 ]; then bad "barley $name exited $rc"
        elif ! gpu_used "$d.log" "$route"; then bad "barley $name did not take the GPU path"
        elif diff -r "$ref" "$d" > /dev/null; then ok "barley $name ($(grep -h 'Reference processed in' "$d.log" | tail -1))"
        else bad "barley $name output differs"; fi
    done
fi

echo; echo "--- routing and path lines, for the record"
for f in "$O"/scan_k25_50000_g2.log "$O"/scan_k25_50000_g2staged.log "$O"/int_k25xk25_g2staged_t16.log; do
    [ -f "$f" ] && { echo "$f:"; grep -E '^\[gpu\] (routing|splitting|pipeline)|^K-mer path' "$f" | head -5 | sed 's/^/    /'; }
done
echo; echo "$checks checks"
[ $status = 0 ] && echo "=== RELEASE GATE PASSED" || echo "=== RELEASE GATE FAILED"
exit $status
