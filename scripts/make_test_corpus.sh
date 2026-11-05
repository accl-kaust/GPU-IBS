#!/bin/bash
# Generate a synthetic regression corpus for validating the FastIBS /
# GPU-IBS pipelines, so the phase gates can be run without production
# plant-genome data.
#
# The reference deliberately contains the edge cases the phase gates
# require: lowercase bases, N runs of length 1, k-1, k and 3k, a contig
# shorter than k, a contig exactly k long, and windows that end mid-contig.
# The KMC database is built from a mutated copy of that reference (SNPs
# plus dropout blocks), so hit/miss runs, gap statistics and variations are
# all non-trivial rather than uniformly present or absent.
#
# Usage:
#   scripts/make_test_corpus.sh <out_dir> [k] [ref_bp] [seed]
# Defaults: k=25, ref_bp=2000000, seed=1
#
# Produces:
#   <out_dir>/refs/test_genome.fasta      -> pass as <referencePath>
#   <out_dir>/db/sample.kmc_pre|.kmc_suf  -> pass as <sourcePath>
#
# Requires: kmc on PATH (conda install -c bioconda kmc), GNU awk (gawk).
# The corpus is a function of the seed AND of gawk's rand(): mawk or busybox
# awk give a different corpus for the same seed, so the script refuses them
# unless AWK is set explicitly. The sha256 of both FASTAs is printed so a
# corpus can be recorded and checked against another machine's.
# Environment (for corpora of billions of bases):
#   KMC_THREADS=4     threads for kmc (-t)
#   KMC_MEM_GB=       RAM cap for kmc (-m); kmc's own default when unset
#   AWK=gawk          awk implementation (anything but gawk changes the corpus)

set -euo pipefail

OUT=${1:?usage: make_test_corpus.sh <out_dir> [k] [ref_bp] [seed]}
K=${2:-25}
REF_BP=${3:-2000000}
SEED=${4:-1}
KMC_THREADS=${KMC_THREADS:-4}
KMC_MEM=${KMC_MEM_GB:+-m${KMC_MEM_GB}}
AWK=${AWK:-gawk}

if ! command -v kmc >/dev/null; then
    echo "error: kmc not found on PATH (conda install -c bioconda kmc)" >&2
    exit 1
fi
if ! command -v "$AWK" >/dev/null; then
    echo "error: $AWK not found (install GNU awk; the recorded corpora were made" >&2
    echo "       with gawk, and another awk gives a different corpus for the same seed)" >&2
    exit 1
fi
echo "awk: $({ "$AWK" --version 2>/dev/null </dev/null || echo "$AWK (version unknown)"; } | sed -n 1p)"

mkdir -p "$OUT/refs" "$OUT/db" "$OUT/kmc_tmp"
REF_FA="$OUT/refs/test_genome.fasta"
SMP_FA="$OUT/sample_reads.fasta"

echo "generating reference (${REF_BP} bp main contig, k=${K}) ..."
"$AWK" -v seed="$SEED" -v k="$K" -v mainlen="$REF_BP" \
    -v ref="$REF_FA" -v smp="$SMP_FA" '
function rb() { return substr("ACGT", int(rand() * 4) + 1, 1) }

# N runs in the flagged contigs: lengths 1, k-1, k and 3k. Positions are
# spaced so each run sits well inside the contig.
function isN(p, len) {
    if (p >= 1000 && p < 1001) return 1
    if (p >= 5000 && p < 5000 + k - 1) return 1
    if (p >= 9000 && p < 9000 + k) return 1
    if (p >= 13000 && p < 13000 + 3 * k) return 1
    return 0
}

# Dropout blocks removed from the sample copy: the corresponding reference
# k-mers are absent from the database, producing gap runs and variations.
function isDrop(p) {
    if (p >= 200000 && p < 205000) return 1
    if (p >= 400000 && p < 402000) return 1
    if (p >= 900000 && p < 900100) return 1
    return 0
}

# Emit one contig into both the reference and the sample copy, 80 bp per
# line, without ever holding the whole sequence in memory.
function gen(name, len, lower, withN,    pos, m, j, c, rc, sc, rline, sline) {
    print ">" name > ref
    print ">" name > smp
    pos = 0
    while (pos < len) {
        m = (len - pos > 80) ? 80 : len - pos
        rline = ""
        sline = ""
        for (j = 0; j < m; j++) {
            c = rb()
            rc = c
            if (withN && isN(pos + j)) rc = "N"
            if (lower) rc = tolower(rc)
            rline = rline rc

            if (!isDrop(pos + j)) {
                sc = c
                if (rand() < 0.01) sc = rb()   # ~1% SNPs
                sline = sline sc
            }
        }
        print rline > ref
        if (length(sline) > 0) print sline > smp
        pos += m
    }
}

BEGIN {
    srand(seed)
    gen("chr1", mainlen, 0, 0)          # plain, with dropout blocks
    gen("chr2_lowercase", 100000, 1, 0) # lowercase bases
    gen("chr3_withN", 60000, 0, 1)      # N runs of 1, k-1, k, 3k
    gen("chr4_short", k - 3, 0, 0)      # shorter than k: zero k-mers
    gen("chr5_exactk", k, 0, 0)         # exactly k: one k-mer
    gen("chr6_mixed", 50000, 1, 1)      # lowercase + N runs
}'
echo "sha256 (record with the corpus):"
sha256sum "$REF_FA" "$SMP_FA" | sed 's/^/  /'

echo "building KMC database (k=${K}) ..."
# shellcheck disable=SC2086
kmc -k"${K}" -fm -ci1 -t"${KMC_THREADS}" ${KMC_MEM} "$SMP_FA" "$OUT/db/sample" "$OUT/kmc_tmp" > "$OUT/kmc.log" 2>&1 \
    || { echo "kmc failed; see $OUT/kmc.log" >&2; exit 1; }

rm -rf "$OUT/kmc_tmp"

echo
echo "corpus ready:"
echo "  reference folder : $OUT/refs"
echo "  database folder  : $OUT/db"
ls -la "$OUT/db"
echo
echo "run with:"
echo "  build/fastibs $OUT/db $OUT/refs <results_dir> 50000"
