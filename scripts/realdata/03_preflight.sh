#!/bin/bash
# Real-data walkthrough, step 3: does this KMC database fit the GPU(s)?
#
# With one device GPU-IBS keeps the whole index resident; with --gpus=N the
# index is split across the N devices by prefix range (Phase 5), each holding
# about 1/N of the suffix and look-up-table areas. Per device:
#   footprint = (stride * N + 8 * bins * 4^p) / SHARDS  (+ ~16 MiB of signature tables)
# N = total k-mers, p = the lut_prefix_length KMC chose, stride = (k-p)/4
# suffix bytes widened to 4 or 8 (the on-disk counter byte is stripped).
# This is the paper's sizing model; on the recorded 2,000,199,169-k-mer
# database (k=25, p=5) it gives 15,266 MiB = 16.01 GB, which is exactly what
# the tool reports at upload.
#
# Usage:
#   bash scripts/realdata/03_preflight.sh <kmc_db_folder_or_prefix> [device_MiB]
#   bash scripts/realdata/03_preflight.sh --from-info <kmc_tools_info.txt> [device_MiB]
# device_MiB defaults to nvidia-smi's memory.total when a GPU is visible,
# otherwise the verdict is given for 40 GB and 80 GB A100s.
# Environment:
#   REQUIRE_FIT=1   exit 2 when footprint + margin exceeds device_MiB
#   MARGIN_GIB=4    headroom for the streamed pipeline's device buffers
#   BINS=512        KMC's number of bins (-n; 512 is the default)
#   SHARDS=1        devices the index is split across (--gpus=N); 1 = whole index
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 1; }
MARGIN_GIB=${MARGIN_GIB:-4}
BINS=${BINS:-512}
SHARDS=${SHARDS:-1}

if [ "$1" = "--from-info" ]; then
    INFO=$(cat "${2:?missing info file}")
    DEV_MIB=${3:-}
    SRC="$2"
else
    SRC=$1
    DEV_MIB=${2:-}
    if [ -d "$SRC" ]; then
        PRE=$(ls "$SRC"/*.kmc_pre 2>/dev/null | head -1)
        [ -n "$PRE" ] || { echo "error: no .kmc_pre in $SRC" >&2; exit 1; }
        SRC=${PRE%.kmc_pre}
    fi
    [ -f "$SRC.kmc_pre" ] || { echo "error: $SRC.kmc_pre not found" >&2; exit 1; }
    KT=$(command -v kmc_tools || true)
    [ -n "$KT" ] || { echo "error: kmc_tools not found on PATH (conda activate gpuibs, or conda install -c bioconda kmc)" >&2; exit 1; }
    INFO=$("$KT" info "$SRC")
fi

if [ -z "$DEV_MIB" ] && command -v nvidia-smi >/dev/null 2>&1; then
    DEV_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
fi

echo "database: $SRC"
echo "$INFO" | sed 's/^/  /'
echo

read -r K N P SIG <<< "$(echo "$INFO" | awk -F: '
    /^k[ \t]*:/            { gsub(/[ \t]/, "", $2); k = $2 }
    /total k-mers/         { gsub(/[ \t]/, "", $2); n = $2 }
    /lut_prefix_len/       { gsub(/[ \t]/, "", $2); p = $2 }
    /signature length/     { gsub(/[ \t]/, "", $2); s = $2 }
    END { print k, n, p, s }')"
[ -n "$K" ] && [ -n "$N" ] && [ -n "$P" ] || { echo "error: could not parse k / total k-mers / lut_prefix_length from kmc_tools info" >&2; exit 1; }

awk -v k="$K" -v n="$N" -v p="$P" -v sig="${SIG:-9}" -v bins="$BINS" -v margin="$MARGIN_GIB" -v dev="$DEV_MIB" -v req="${REQUIRE_FIT:-0}" -v shards="$SHARDS" '
function gib(b) { return b / 1073741824 }
function verdict(name, mib,   ok) {
    ok = (need_mib <= mib)
    printf "  %-28s %s (device %d MiB, need %d MiB incl. %s GiB margin)\n", name, ok ? "FITS" : "DOES NOT FIT", mib, need_mib, margin
    return ok
}
BEGIN {
    if ((k - p) % 4 != 0) print "warning: (k - p) not divisible by 4; KMC databases always are -- check the info parse"
    sufbytes = (k - p) / 4
    stride = (sufbytes <= 4) ? 4 : 8
    lut = 8 * bins * (4 ^ p)
    # signature map + m-mer normalization table: one uint32 per m-mer each (2 MiB at m=9)
    sigtab = 2 * 4 * (4 ^ sig)
    if (shards < 1) shards = 1
    index_b = (stride * n + lut) / shards + sigtab
    need_mib = int(index_b / 1048576 + margin * 1024 + 0.5)
    if (shards > 1) printf "index split across %d devices (--gpus=%d): the figures below are PER DEVICE\n", shards, shards
    printf "k=%d  total k-mers N=%.0f  lut_prefix_length p=%d  signature length m=%d  bins=%d\n", k, n, p, sig, bins
    printf "suffix bytes on disk=%d (+1 counter)  device stride=%d B/k-mer\n", sufbytes, stride
    printf "device index = %.0f x %d B + LUT %.2f GiB + tables %.0f MiB  = %.2f GiB = %.2f GB (%.0f MiB)\n", n, stride, gib(lut), sigtab / 1048576, gib(index_b), index_b / 1e9, index_b / 1048576
    printf "host RAM while loading: image %.0f GiB + page cache of .kmc_suf %.0f GiB  -> --mem >= %dG per process (x2 images for the two-process run: >= %dG)\n",
        gib(stride * n), gib((sufbytes + 1) * n), int(gib(stride * n) + gib((sufbytes + 1) * n) + 12), int(2 * gib(stride * n) + gib((sufbytes + 1) * n) + 12)
    print ""
    fits = 1
    if (dev != "") {
        fits = verdict("this node'"'"'s GPU", dev + 0)
    } else {
        verdict("A100 40 GB (40960 MiB)", 40960)
        verdict("A100 80 GB (81920 MiB)", 81920)
        fits = (need_mib <= 81920)
        print "  (no GPU visible here: choose the node from the two lines above)"
    }
    if (!fits && req == "1") exit 2
}'
