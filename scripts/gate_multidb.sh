#!/bin/bash
# Bit-exactness gate of panel mode (docs/MULTIDB_SCAN.md section 9): for
# N in {2, 3, 5} databases of a panel corpus and W in {10000, 50000}, the N
# TSVs of one panel run must be byte-identical to the N TSVs of N single
# runs; a forced-rounds pass (--resident=2) checks the round logic. Runs
# whatever path the binary takes (CPU-only build, CUDA build without a
# device, or the GPU on a GPU node) -- pass --cpu / --gpu to pin it.
#
# Usage:
#   scripts/gate_multidb.sh <fastibs binary> <panel corpus dir> [extra fastibs options...]
#   (the corpus from scripts/make_test_panel.sh <dir> 5 [k])
#   NS="2 3 5"  WS="10000 50000"  KEEP=1 (keep the work dir)
set -euo pipefail

BIN=${1:?usage: gate_multidb.sh <fastibs> <panel_corpus_dir> [options...]}
CORPUS=${2:?missing <panel_corpus_dir>}
shift 2
NS=${NS:-"2 3 5"}
WS=${WS:-"10000 50000"}
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/gate_multidb.XXXXXX")}
REFS=$CORPUS/refs
fail=0
checks=0

ndb=$(ls -d "$CORPUS"/db[0-9]* | wc -l)
echo "=== panel gate: $BIN, corpus $CORPUS ($ndb databases), options: $*"
echo "    work dir $WORK"

# Single runs, once per (database, W): the reference outputs.
for W in $WS; do
    for ((i = 1; i <= ndb; i++)); do
        d=$WORK/single_${i}_$W
        mkdir -p "$d"
        "$BIN" "$CORPUS/db$i" "$REFS" "$d" "$W" "$@" > "$d/log" 2>&1 \
            || { echo "FAIL: single run db$i W=$W exited $? (see $d/log)"; fail=1; }
    done
done

compare () { # label W N panel_dir
    local label=$1 W=$2 N=$3 pd=$4
    for ((i = 1; i <= N; i++)); do
        local f="db${i}_v_test_${W}.tsv"
        checks=$((checks + 1))
        if [ ! -f "$pd/$f" ]; then
            echo "FAIL: $label W=$W N=$N: missing $pd/$f"; fail=1
        elif ! cmp -s "$WORK/single_${i}_$W/$f" "$pd/$f"; then
            echo "FAIL: $label W=$W N=$N: $f differs from the single run"; fail=1
            diff "$WORK/single_${i}_$W/$f" "$pd/$f" | head -5
        fi
    done
}

for W in $WS; do
    for N in $NS; do
        if [ "$N" -gt "$ndb" ]; then echo "skip N=$N (corpus has $ndb)"; continue; fi
        dbs=()
        for ((i = 2; i <= N; i++)); do dbs+=("--db=$CORPUS/db$i"); done
        # (a) panel in one round
        pd=$WORK/panel_${N}_$W; mkdir -p "$pd"
        "$BIN" "$CORPUS/db1" "$REFS" "$pd" "$W" "${dbs[@]}" "$@" > "$pd/log" 2>&1 \
            || { echo "FAIL: panel N=$N W=$W exited $? (see $pd/log)"; fail=1; }
        compare panel "$W" "$N" "$pd"
        echo "  N=$N W=$W: $(grep -o '\[panel\] reference done.*' "$pd/log" | head -1)"
        # (b) the same through --db-list and in rounds of 2
        if [ "$N" -ge 3 ]; then
            list=$pd/list.txt; : > "$list"
            for ((i = 2; i <= N; i++)); do echo "$CORPUS/db$i" >> "$list"; done
            pr=$WORK/rounds_${N}_$W; mkdir -p "$pr"
            "$BIN" "$CORPUS/db1" "$REFS" "$pr" "$W" "--db-list=$list" --resident=2 "$@" > "$pr/log" 2>&1 \
                || { echo "FAIL: panel rounds N=$N W=$W exited $? (see $pr/log)"; fail=1; }
            compare rounds "$W" "$N" "$pr"
            echo "  N=$N W=$W --resident=2: $(grep -o '\[panel\] [0-9]* databases in [0-9]* rounds.*' "$pr/log" | head -1)"
        fi
    done
done

# Two references in one folder: the panel keeps a single round resident
# from the first reference to the second and releases rounds otherwise;
# both files of every database must match the single runs over the same
# folder.
W=${WS%% *}
N=3
if [ "$N" -le "$ndb" ]; then
    refs2=$WORK/refs2; mkdir -p "$refs2"
    ln -sf "$(readlink -f "$REFS/test_genome.fasta")" "$refs2/test_genome.fasta"
    ln -sf "$(readlink -f "$REFS/test_genome.fasta")" "$refs2/other_genome.fasta"
    for ((i = 1; i <= N; i++)); do
        d=$WORK/single2_${i}; mkdir -p "$d"
        "$BIN" "$CORPUS/db$i" "$refs2" "$d" "$W" "$@" > "$d/log" 2>&1 || { echo "FAIL: single db$i two references"; fail=1; }
    done
    for res in 0 2; do
        pd=$WORK/panel2_res$res; mkdir -p "$pd"
        extra=(); [ "$res" != 0 ] && extra=("--resident=$res")
        "$BIN" "$CORPUS/db1" "$refs2" "$pd" "$W" "--db=$CORPUS/db2" "--db=$CORPUS/db3" "${extra[@]}" "$@" > "$pd/log" 2>&1 \
            || { echo "FAIL: panel two references resident=$res exited $?"; fail=1; }
        for ((i = 1; i <= N; i++)); do
            for r in test other; do
                f="db${i}_v_${r}_${W}.tsv"; checks=$((checks + 1))
                cmp -s "$WORK/single2_${i}/$f" "$pd/$f" || { echo "FAIL: two references, resident=$res: $f differs or is missing"; fail=1; }
            done
        done
        echo "  two references, N=$N, resident=$res: $(grep -c '^\[panel\] reference done' "$pd/log") reference passes"
    done
fi

# Existing files are skipped per database, and a panel run with every file
# present does nothing. The output goes to a file before it is searched:
# piping it into grep -q under pipefail fails at random, because grep exits
# at the first match and the next line fastibs writes kills it with SIGPIPE.
pd=$WORK/panel_2_${WS%% *}
touch "$pd/skip.stamp"
rc=0
"$BIN" "$CORPUS/db1" "$REFS" "$pd" "${WS%% *}" "--db=$CORPUS/db2" "$@" > "$pd/skip.log" 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "FAIL: panel rerun over existing files exited $rc (see $pd/skip.log)"; fail=1
elif [ "$(grep -c 'already exists' "$pd/skip.log")" -lt 2 ]; then
    echo "FAIL: skip-if-exists not honoured in panel mode (see $pd/skip.log)"; fail=1
elif [ -n "$(find "$pd" -name '*.tsv' -newer "$pd/skip.stamp")" ]; then
    echo "FAIL: panel rerun rewrote existing files"; fail=1
fi

if [ "$fail" = 0 ]; then
    echo "=== PASS: $checks TSV comparisons identical (NS=$NS, WS=$WS)"
    [ "${KEEP:-0}" = 1 ] || rm -rf "$WORK"
else
    echo "=== FAIL (work dir kept: $WORK)"
fi
exit $fail
