#!/bin/bash
# Real-data walkthrough, step 1: fetch the reads and the reference, lay out
# the FastIBS data volume, and split the reference into two balanced halves
# for the two-GPU run.
#
# Dataset (default): ENA run ERR6098905 -- Hordeum vulgare (barley), genotype
# "40" of the IPK SHAPE panel (study PRJEB45512, "Raw sequence data of 47
# diverse barley (Hordeum vulgare L.) genotypes", ~50x Illumina, first public
# 2023-11-17). Illumina NovaSeq 6000, 2x250 bp, 1,416,842,440 reads,
# 212,985,590,007 bases, 99.67 GB as two gzipped FASTQ files:
#   ERR6098905_1.fastq.gz  48,867,491,666 B  md5 69d9e957635095b300b8221c8326caed
#   ERR6098905_2.fastq.gz  50,807,061,323 B  md5 de8eb14ddd92fc83a383e3270dfffac9
# (sizes and md5s are re-read from ENA at run time, so any run accession works).
# Reference: MorexV3 pseudomolecules, chromosomes 1H..7H (~4.2 Gbp), from
# Ensembl Plants, verified against the directory's CHECKSUMS file.
#
# Result layout (what run.sh / fastibs expect, plus the two GPU halves):
#   <data_root>/reads/<RUN>_{1,2}.fastq.gz, <RUN>.files   (KMC input list)
#   <data_root>/kmc_sets/                                 (filled by step 2)
#   <data_root>/reference/MorexV3_<chr>_genome.fasta      (one file per chromosome)
#   <data_root>/reference_gpu0, reference_gpu1            (relative symlinks, balanced by size)
#   <data_root>/FastIBS_runs/                             (results)
#
# Usage:
#   bash scripts/realdata/01_get_data.sh <data_root> [ENA_run]
#   STEP=all|reads|ref|split   CONNS=8 (aria2 connections)   REF_NAME=MorexV3
# Run where the internet is reachable (login or data-transfer node), under
# tmux or nohup: the reads alone are 100 GB. Re-running resumes and skips
# what is already verified.
set -euo pipefail

DATA=${1:?usage: 01_get_data.sh <data_root> [ENA_run]}
RUN=${2:-ERR6098905}
STEP=${STEP:-all}
CONNS=${CONNS:-8}
REF_NAME=${REF_NAME:-MorexV3}
CHROMS=${CHROMS:-"1H 2H 3H 4H 5H 6H 7H"}
ENA_API="https://www.ebi.ac.uk/ena/portal/api/filereport"
ENSEMBL_DIR=${ENSEMBL_DIR:-https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/hordeum_vulgare/dna}
REF_FILE_PATTERN=${REF_FILE_PATTERN:-Hordeum_vulgare.MorexV3_pseudomolecules_assembly.dna.primary_assembly.%s.fa.gz}

mkdir -p "$DATA"/reads "$DATA"/kmc_sets "$DATA"/reference "$DATA"/FastIBS_runs
DATA=$(cd "$DATA" && pwd)

# aria2c (multi-connection, resumable) when available -- on PATH, or installed
# in the conda env even if that env is not active in this shell; wget otherwise.
ARIA2=$(command -v aria2c 2>/dev/null || true)
if [ -z "$ARIA2" ] && command -v conda >/dev/null 2>&1; then
    cand="$(conda info --base 2>/dev/null)/envs/${CONDA_ENV:-gpuibs}/bin/aria2c"
    [ -x "$cand" ] && ARIA2=$cand
fi
echo "downloader: ${ARIA2:-wget (install aria2 into the conda env for a faster multi-connection download)}"

fetch () { # url dest  (resumable; an orphaned partial file without aria2 control file is restarted)
    local url=$1 dest=$2
    if [ -n "$ARIA2" ]; then
        "$ARIA2" -c -x "$CONNS" -s "$CONNS" -k 16M --file-allocation=none \
            --allow-overwrite=true --auto-file-renaming=false \
            --console-log-level=warn --summary-interval=120 \
            -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
    else
        wget -c -q --show-progress -O "$dest" "$url"
    fi
}

get_reads () {
    echo "=== reads: ENA run ${RUN} ==="
    local tsv
    tsv=$(curl -sS --fail --retry 5 \
        "${ENA_API}?accession=${RUN}&result=read_run&fields=fastq_bytes,fastq_md5,fastq_ftp&format=tsv")
    [ "$(printf '%s\n' "$tsv" | grep -c .)" -ge 2 ] || { echo "error: ENA returned no record for ${RUN}" >&2; exit 1; }
    # ENA prepends run_accession to the requested fields, so pick the columns
    # by header name rather than by position.
    col () { printf '%s\n' "$tsv" | awk -F'\t' -v name="$1" 'NR==1{for(i=1;i<=NF;i++) if($i==name) c=i} NR==2{if(c) print $c}'; }
    local bytes md5s ftps
    bytes=$(col fastq_bytes); md5s=$(col fastq_md5); ftps=$(col fastq_ftp)
    [ -n "$bytes" ] && [ -n "$md5s" ] && [ -n "$ftps" ] \
        || { echo "error: could not parse the ENA file report for ${RUN}:" >&2; printf '%s\n' "$tsv" >&2; exit 1; }
    local -a BYTES MD5S FTPS
    IFS=';' read -ra BYTES <<< "$bytes"
    IFS=';' read -ra MD5S <<< "$md5s"
    IFS=';' read -ra FTPS <<< "$ftps"
    [ "${#FTPS[@]}" -gt 0 ] || { echo "error: no FASTQ files listed for ${RUN}" >&2; exit 1; }
    # The KMC input list is assembled in a .part file and moved into place only
    # when every file is complete and verified, so a KMC job submitted while the
    # download is still running finds no list rather than a partial one.
    local list="$DATA/reads/${RUN}.files"
    rm -f "$list"
    : > "$list.part"
    local total=0 i
    for i in "${!FTPS[@]}"; do
        local url="https://${FTPS[$i]}" f="$DATA/reads/$(basename "${FTPS[$i]}")"
        total=$((total + BYTES[i]))
        if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "${BYTES[$i]}" ]; then
            echo "have $(basename "$f") (${BYTES[$i]} bytes)"
        else
            echo "downloading $(basename "$f") (${BYTES[$i]} bytes) from $url ..."
            fetch "$url" "$f"
            [ "$(stat -c %s "$f")" = "${BYTES[$i]}" ] \
                || { echo "error: size mismatch for $f (have $(stat -c %s "$f"), ENA says ${BYTES[$i]})" >&2; exit 1; }
        fi
        if [ -f "$f.md5.ok" ] && grep -q "^${MD5S[$i]} " "$f.md5.ok"; then
            echo "  md5 verified earlier"
        else
            echo "  verifying md5 ..."
            local got; got=$(md5sum "$f" | cut -d' ' -f1)
            [ "$got" = "${MD5S[$i]}" ] \
                || { echo "error: md5 mismatch for $f (have $got, ENA says ${MD5S[$i]}); delete it and re-run" >&2; exit 1; }
            echo "${MD5S[$i]}  $(basename "$f")" > "$f.md5.ok"
        fi
        echo "$f" >> "$list.part"
    done
    mv "$list.part" "$list"
    echo "reads complete: $((total / 1000000000)) GB in ${#FTPS[@]} files; list in $list"
}

get_ref () {
    echo "=== reference: ${REF_NAME} chromosomes ${CHROMS} from ${ENSEMBL_DIR} ==="
    curl -sS --fail --retry 5 "${ENSEMBL_DIR}/CHECKSUMS" -o "$DATA/reference/CHECKSUMS"
    local c
    for c in $CHROMS; do
        # shellcheck disable=SC2059
        local gz; gz=$(printf "$REF_FILE_PATTERN" "$c")
        # fastibs only picks up files whose extension is exactly .fasta, and it
        # strips a trailing "_genome" from the stem when naming its output.
        local out="$DATA/reference/${REF_NAME}_${c}_genome.fasta"
        if [ -s "$out" ]; then
            echo "have $(basename "$out")"
            continue
        fi
        fetch "${ENSEMBL_DIR}/${gz}" "$DATA/reference/${gz}"
        # Ensembl CHECKSUMS lines are BSD `sum` output: "<sum> <blocks> <file>"
        local want have
        want=$(awk -v f="$gz" '$3==f{print $1" "$2}' "$DATA/reference/CHECKSUMS")
        have=$(cd "$DATA/reference" && sum "$gz" | awk '{print $1" "$2}')
        [ -n "$want" ] || { echo "error: $gz not in CHECKSUMS (file renamed upstream?)" >&2; exit 1; }
        [ "$want" = "$have" ] || { echo "error: checksum mismatch for $gz (want '$want', have '$have')" >&2; exit 1; }
        gunzip -c "$DATA/reference/${gz}" > "$out.part" && mv "$out.part" "$out"
        rm -f "$DATA/reference/${gz}"
        echo "wrote $(basename "$out") ($(stat -c %s "$out") bytes, header: $(head -c 60 "$out" | head -1))"
    done
}

split_ref () {
    echo "=== splitting the reference into two halves for two GPU processes ==="
    rm -rf "$DATA/reference_gpu0" "$DATA/reference_gpu1"
    mkdir -p "$DATA/reference_gpu0" "$DATA/reference_gpu1"
    local listing
    listing=$(find "$DATA/reference" -maxdepth 1 -name '*.fasta' -printf '%s %f\n' | sort -rn)
    [ "$(printf '%s\n' "$listing" | grep -c .)" -ge 2 ] \
        || { echo "error: fewer than two .fasta files in $DATA/reference" >&2; exit 1; }
    # Two-way partition minimizing the larger half: exact (all 2^(n-1)
    # splits) for up to 16 files, greedy largest-first beyond that.
    local plan
    plan=$(printf '%s\n' "$listing" | awk '
        { n++; s[n] = $1; f[n] = $2; tot += $1 }
        END {
            if (n <= 16) {
                best = tot + 1
                for (m = 0; m < 2 ^ (n - 1); m++) {          # file n stays on side 0: no mirrored duplicates
                    a = 0
                    for (i = 1; i <= n; i++) if (int(m / 2 ^ (i - 1)) % 2) a += s[i]
                    mx = (a > tot - a) ? a : tot - a
                    if (mx < best) { best = mx; bm = m }
                }
            } else {
                a = 0; b = 0; bm = 0
                for (i = 1; i <= n; i++) if (a <= b) { bm += 2 ^ (i - 1); a += s[i] } else b += s[i]
            }
            for (i = 1; i <= n; i++) print (int(bm / 2 ^ (i - 1)) % 2), s[i], f[i]
        }')
    local side size base
    while read -r side size base; do
        # relative symlinks, so the folders also work inside the container (bound at /project/data)
        ln -s "../reference/$base" "$DATA/reference_gpu$side/$base"
    done <<< "$plan"
    for side in 0 1; do
        printf '%s\n' "$plan" | awk -v side="$side" -v d="reference_gpu$side" '
            $1 == side { s += $2; names = names " " $3 }
            END { gsub(/_genome\.fasta/, "", names); printf "%s (%d MB):%s\n", d, s / 1000000, names }'
    done
    printf '%s\n' "$plan" | awk '{ s[$1] += $2; tot += $2 }
        END { m = (s[0] > s[1]) ? s[0] : s[1]
              printf "larger half = %.1f%% of the reference -> ideal two-process speedup on the processing part = %.2fx\n", 100 * m / tot, tot / m }'
}

case "$STEP" in
    all)   get_reads; get_ref; split_ref ;;
    reads) get_reads ;;
    ref)   get_ref ;;
    split) split_ref ;;
    *) echo "error: STEP must be all|reads|ref|split" >&2; exit 1 ;;
esac

echo
echo "data volume: $DATA"
du -sh "$DATA"/reads "$DATA"/reference 2>/dev/null || true
echo "next: sbatch scripts/realdata/02_build_kmc.sbatch $DATA $RUN"
