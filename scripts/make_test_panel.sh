#!/bin/bash
# Synthetic PANEL corpus for the multi-database scan (docs/MULTIDB_SCAN.md):
# ONE reference and N KMC databases, each built from a differently-seeded
# mutated copy of that reference (its own ~1% SNPs and its own dropout
# blocks), so the N hit patterns differ while the reference is shared.
# Database 1 and the reference are exactly make_test_corpus.sh's (same
# seed, same edge cases: lowercase, N runs, a contig shorter than k, one of
# exactly k); databases 2..N mutate the reference with seed+i.
#
# Usage:
#   scripts/make_test_panel.sh <out_dir> <N> [k] [ref_bp] [seed]
# Defaults: k=25, ref_bp=2000000, seed=1
#
# Produces:
#   <out_dir>/refs/test_genome.fasta           -> <referencePath>
#   <out_dir>/db1/sample.kmc_pre|.kmc_suf      -> <sourcePath>
#   <out_dir>/db2 ... <out_dir>/dbN            -> --db=<out_dir>/db2 ...
#   <out_dir>/dbs.txt                          -> --db-list (db2..dbN)
#
# Requires: kmc on PATH, GNU awk (gawk; AWK=... overrides, but another awk
# gives different databases for the same seed: see make_test_corpus.sh).

set -euo pipefail

OUT=${1:?usage: make_test_panel.sh <out_dir> <N> [k] [ref_bp] [seed]}
N=${2:?missing <N>}
K=${3:-25}
REF_BP=${4:-2000000}
SEED=${5:-1}
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
AWK=${AWK:-gawk}

if ! command -v kmc >/dev/null; then
    echo "error: kmc not found on PATH (conda install -c bioconda kmc)" >&2
    exit 1
fi
if ! command -v "$AWK" >/dev/null; then
    echo "error: $AWK not found (install GNU awk; another awk gives a different panel)" >&2
    exit 1
fi
export AWK

# Reference + database 1: the single-corpus generator, verbatim.
mkdir -p "$OUT"
bash "$HERE/make_test_corpus.sh" "$OUT" "$K" "$REF_BP" "$SEED" > "$OUT/corpus.log"
grep -E '^awk:|refs/test_genome\.fasta$' "$OUT/corpus.log" | sed 's/^ *\([0-9a-f]\{64\}\)/  sha256 \1/'
mv "$OUT/db" "$OUT/db1"
rm -f "$OUT/sample_reads.fasta"
: > "$OUT/dbs.txt"

REF_FA="$OUT/refs/test_genome.fasta"
for ((i = 2; i <= N; i++)); do
    echo "database $i: mutating the reference with seed $((SEED + i)) ..."
    SMP_FA="$OUT/sample_$i.fasta"
    # Per database: its own SNP positions (rate ~1%) and three dropout
    # blocks at seed-dependent offsets, applied to the upper-cased
    # reference with N runs left out of the sample (as the generator does
    # for database 1: N is not a base of the sample).
    "$AWK" -v seed="$((SEED + i))" -v smp="$SMP_FA" '
    function rb() { return substr("ACGT", int(rand() * 4) + 1, 1) }
    BEGIN {
        srand(seed)
        d1 = 100000 + int(rand() * 1500000); d2 = d1 + 300000 + int(rand() * 200000); d3 = 50000 + int(rand() * 40000)
    }
    /^>/ { if (name != "") flush(); name = $0; pos = 0; print name > smp; next }
    {
        line = toupper($0)
        out = ""
        for (j = 1; j <= length(line); j++) {
            c = substr(line, j, 1); p = pos + j - 1
            if (c != "A" && c != "C" && c != "G" && c != "T") continue
            if (name == ">chr1" && ((p >= d1 && p < d1 + 4000) || (p >= d2 && p < d2 + 1500))) continue
            if (name != ">chr1" && p >= d3 && p < d3 + 200) continue
            if (rand() < 0.01) c = rb()
            out = out c
            if (length(out) == 80) { print out > smp; out = "" }
        }
        if (length(out) > 0) print out > smp
        pos += length(line)
    }
    function flush() {}
    ' "$REF_FA"
    sha256sum "$SMP_FA" | sed "s/^/  sha256 /"
    mkdir -p "$OUT/db$i" "$OUT/kmc_tmp"
    kmc -k"${K}" -fm -ci1 -t4 "$SMP_FA" "$OUT/db$i/sample" "$OUT/kmc_tmp" > "$OUT/kmc_$i.log" 2>&1 \
        || { echo "kmc failed for database $i; see $OUT/kmc_$i.log" >&2; exit 1; }
    rm -rf "$OUT/kmc_tmp" "$SMP_FA"
    echo "$OUT/db$i" >> "$OUT/dbs.txt"
done

echo
echo "panel corpus ready: reference $OUT/refs, databases $OUT/db1 .. $OUT/db$N (list: $OUT/dbs.txt)"
echo "run with:"
echo "  build/fastibs $OUT/db1 $OUT/refs <results_dir> 50000 --db-list=$OUT/dbs.txt"
