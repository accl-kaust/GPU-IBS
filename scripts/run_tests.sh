#!/bin/bash
# One command for the whole host-side test suite:
#   1. builds every regression test in tests/ (the g++ lines of their headers)
#   2. generates the synthetic corpora they read (scripts/make_test_corpus.sh,
#      scripts/make_test_panel.sh; needs kmc and GNU awk)
#   3. runs the tests
#   4. when the tools are built (bin/ or BIN_DIR), checks them end to end:
#      fastibs, fastibsmapper and KDBIntersect against the original FastIBS
#      code path (FASTIBS_LEGACY=1) at k=25, k=31 and k=35, panel mode
#      against single runs, and -- when a GPU is visible and the tools were
#      built with CUDA -- the GPU path against the CPU path, byte for byte.
#
# Usage:
#   bash scripts/run_tests.sh [work_dir]          (default: ./test_work)
# Environment:
#   BIN_DIR=bin       where fastibs, fastibsmapper, KDBIntersect are
#                     (step 4 is skipped when they are missing)
#   CXX=g++           compiler for the tests (C++20, zlib, Boost headers)
#   SKIP_TOOLS=1      run steps 1-3 only
#   GPU=0             never use the GPU in step 4, even when one is visible
#
# Exit status 0 when every check passed. The multi-GPU configurations are
# covered by the on-GPU gates (scripts/slurm_*.sbatch, scripts/ibex/50_release_gate.sbatch).
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

W=$(readlink -m "${1:-test_work}")
BIN_DIR=$(readlink -m "${BIN_DIR:-bin}")
CXX=${CXX:-g++}
mkdir -p "$W/tests" "$W/logs"

passed=0; failed=0
pass () { passed=$((passed+1)); echo "  PASS  $*"; }
fail () { failed=$((failed+1)); echo "  FAIL  $*"; }
check () { local rc=$1; shift; if [ "$rc" = 0 ]; then pass "$@"; else fail "$@ (exit $rc)"; fi; }
section () { echo; echo "== $*"; }

for t in kmc "${AWK:-gawk}" "$CXX"; do
    command -v "$t" > /dev/null || { echo "error: $t not found on PATH (kmc: conda install -c bioconda kmc; gawk: your package manager)" >&2; exit 2; }
done

# Headers and libraries of an active conda environment (Boost, zlib), as compile.sh does
EXTRA=()
if [ -n "${CONDA_PREFIX:-}" ]; then
    EXTRA=(-I"$CONDA_PREFIX/include" -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib")
fi

section "building the regression tests ($CXX)"
API=(KMC/kmc_api/kmc_file.cpp KMC/kmc_api/kmer_api.cpp KMC/kmc_api/mmer.cpp)
DEFS=(-DBOOST_ALLOW_DEPRECATED_HEADERS -DBOOST_BIND_GLOBAL_PLACEHOLDERS -DBOOST_TIMER_ENABLE_DEPRECATED)
for src in tests/*.cpp; do
    name=$(basename "$src" .cpp)
    "$CXX" -O2 -std=c++2a "${DEFS[@]}" "$src" "${API[@]}" "${EXTRA[@]}" -lz -lpthread -o "$W/tests/$name" \
        > "$W/logs/build_$name.log" 2>&1
    check $? "build $name"
done

section "synthetic corpora (in $W)"
corpus () {
    local dir=$1; shift
    if [ -f "$W/$dir/db/sample.kmc_pre" ]; then pass "corpus $dir (kept)"; return; fi
    bash scripts/make_test_corpus.sh "$W/$dir" "$@" > "$W/logs/corpus_$dir.log" 2>&1
    check $? "corpus $dir (k=$1)"
}
corpus k25 25
corpus k25b 25 2000000 2
corpus k31 31
if [ ! -f "$W/k35/db/sample.kmc_pre" ]; then
    mkdir -p "$W/k35/db" "$W/k35/tmp"
    kmc -k35 -ci1 -fm -t4 "$W/k25/sample_reads.fasta" "$W/k35/db/sample" "$W/k35/tmp" > "$W/logs/corpus_k35.log" 2>&1
    check $? "corpus k35 (k > 32)"
    rm -rf "$W/k35/tmp"
fi
if [ ! -d "$W/panel/db5" ]; then
    bash scripts/make_test_panel.sh "$W/panel" 5 25 > "$W/logs/corpus_panel.log" 2>&1
    check $? "panel corpus (5 databases)"
fi

section "regression tests"
nrun=0
run () {
    local name=$1; shift
    nrun=$((nrun+1))
    [ -x "$W/tests/$name" ] || { fail "$name (not built)"; return; }
    # phase7_test checks the switch to the original code path itself, so
    # none of the tests may inherit FASTIBS_LEGACY
    env -u FASTIBS_LEGACY "$W/tests/$name" "$@" > "$W/logs/run_${nrun}_$name.log" 2>&1
    check $? "$name ${*//$W\//}"
}
P=$W/panel
run phase0_test
for c in k25 k31; do
    run phase1_test "$W/$c/db/sample"
    run phase2_test "$W/$c/db/sample"
    run phase3_test "$W/$c/db/sample"
done
run phase4_test "$W/k25/db/sample" "$W/k25b/db/sample"
run phase6_test "$P/db1/sample" "$P/db2/sample" "$P/db3/sample" "$P/db4/sample" "$P/db5/sample"
run phase7_test "$W/k25/db/sample" "$W/k35/db/sample"
run load_bench "$W/k25/db/sample" 0

if [ "${SKIP_TOOLS:-0}" = 1 ] || [ ! -x "$BIN_DIR/fastibs" ]; then
    echo; echo "(tools not checked: ${SKIP_TOOLS:+SKIP_TOOLS=1}${SKIP_TOOLS:-no $BIN_DIR/fastibs; build with compile.sh or set BIN_DIR})"
else
    O=$W/out; rm -rf "$O"; mkdir -p "$O"
    # The GPU checks run when a trial --gpu run takes the GPU path (a CUDA
    # build and a usable device); a CPU-only build prints no [gpu] lines,
    # a CUDA build without a device prints the fallback line.
    gpu=0
    if [ "${GPU:-1}" != 0 ]; then
        mkdir -p "$O/probe"
        "$BIN_DIR/fastibs" "$W/k25/db" "$W/k25/refs" "$O/probe" 50000 --gpu > "$O/probe.log" 2>&1
        if grep -q '^\[gpu\] index read' "$O/probe.log" && ! grep -q 'falling back to CPU' "$O/probe.log"; then
            gpu=1
        fi
        rm -rf "$O/probe"
    fi
    section "tools in $BIN_DIR against the original FastIBS code path$([ $gpu = 1 ] && echo ', GPU against CPU' || echo ' (no GPU used: CPU-only build or no usable device)')"
    same () { diff -r "$1" "$2" > /dev/null; }
    for c in k25 k31 k35; do
        refs=$W/${c/k35/k25}/refs
        for tool in fastibs fastibsmapper; do
            args=(50000); [ $tool = fastibsmapper ] && args=()
            o=$O/${tool}_$c; mkdir -p "$o/cpu" "$o/legacy"
            "$BIN_DIR/$tool" "$W/$c/db" "$refs" "$o/cpu" "${args[@]}" --cpu > "$o/cpu.log" 2>&1
            check $? "$tool $c (CPU path)"
            FASTIBS_LEGACY=1 "$BIN_DIR/$tool" "$W/$c/db" "$refs" "$o/legacy" "${args[@]}" --cpu > "$o/legacy.log" 2>&1
            same "$o/cpu" "$o/legacy"; check $? "$tool $c CPU path == original"
            if [ $gpu = 1 ]; then
                mkdir -p "$o/gpu"
                "$BIN_DIR/$tool" "$W/$c/db" "$refs" "$o/gpu" "${args[@]}" --gpu > "$o/gpu.log" 2>&1
                if grep -q 'no CUDA device\|falling back to CPU' "$o/gpu.log"; then fail "$tool $c --gpu fell back to the CPU"
                else same "$o/cpu" "$o/gpu"; check $? "$tool $c GPU path == CPU path"; fi
            fi
        done
    done
    # panel/db1 x panel/db2: two samples of one reference (~1.5 M shared
    # k-mers); k25 x k25b: two unrelated genomes (none shared)
    for pair in "panel/db1 panel/db2" "k25/db k25b/db" "k25/db k25/db" "k31/db k31/db"; do
        set -- $pair
        tag=$(echo "$1_$2" | tr / -)
        want=$(FASTIBS_LEGACY=1 FASTIBS_GPU=0 "$BIN_DIR/KDBIntersect" "$W/$1" "$W/$2" 2> /dev/null)
        got=$(FASTIBS_GPU=0 "$BIN_DIR/KDBIntersect" "$W/$1" "$W/$2" 2> "$O/int_$tag.log")
        [ -n "$want" ] && [ "$got" = "$want" ]; check $? "KDBIntersect $1 x $2 = $got (original: $want)"
        if [ $gpu = 1 ]; then
            got=$("$BIN_DIR/KDBIntersect" "$W/$1" "$W/$2" 2> "$O/int_${tag}_gpu.log")
            [ "$got" = "$want" ]; check $? "KDBIntersect $1 x $2 on the GPU = $got"
        fi
    done
    for mode in --cpu $([ $gpu = 1 ] && echo --gpu); do
        WORK=$O/panel$mode bash scripts/gate_multidb.sh "$BIN_DIR/fastibs" "$W/panel" $mode > "$O/panel$mode.log" 2>&1
        check $? "panel mode $mode: $(grep -o '[0-9]* TSV comparisons identical' "$O/panel$mode.log" || echo 'see log')"
    done
    "$BIN_DIR/fastibs" "$W/k25/db" "$W/k25/refs" "$O/bad" abc > "$O/bad.log" 2>&1
    [ $? = 1 ] && grep -q 'Invalid <windowSize>' "$O/bad.log"; check $? "fastibs rejects a bad window size with exit status 1"
    "$BIN_DIR/KDBIntersect" "$W/k25/db" "$W/k31/db" > "$O/kmix.log" 2>&1
    [ $? = 1 ] && grep -q 'do not match' "$O/kmix.log"; check $? "KDBIntersect rejects databases of different k"
    for t in fastibs fastibsmapper KDBIntersect; do
        v=$("$BIN_DIR/$t" --version 2>&1); rc=$?
        [ $rc = 0 ] && [[ "$v" == "$t (GPU-IBS) "* ]]; check $? "$t --version: $v"
    done
fi

echo
echo "$passed passed, $failed failed  (logs: $W/logs, outputs: $W/out)"
[ "$failed" = 0 ]
