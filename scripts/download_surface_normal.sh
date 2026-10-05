#!/usr/bin/env bash
# Download the surface-normal test sets packaged by Marigold into raw/surface_normal/:
#   NYUv2 test (654), iBims-1 (100), DIODE val (325 indoors used for testing,
#   446 outdoor used only to calibrate the RGB->GT axis convention).
# Same splits, GT normals and valid masks as the DSINE / Marigold / StableNormal
# baselines in RINO (arXiv:2607.12450) Table 3.
# Source: https://github.com/prs-eth/Marigold#-evaluation-on-test-datasets
#
# Usage: scripts/download_surface_normal.sh [target_dir]      (default raw/surface_normal)
#        KEEP_ZIP=1 scripts/download_surface_normal.sh         keep the 21 GB zip afterwards
# The download resumes if interrupted; rerunning after success only re-verifies.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-raw/surface_normal}"
ZIP_URL="${ZIP_URL:-https://share.phys.ethz.ch/~pf/bingkedata/marigold/marigold_normals/evaluation_dataset.zip}"
# Split lists pinned to a Marigold commit so the test sets never drift.
SPLITS="${SPLITS:-https://raw.githubusercontent.com/prs-eth/Marigold/2bfbdeae5d10a50b71f1ba20c865d46e480f6010/data_split}"
ZIP="$TARGET/evaluation_dataset.zip"

# dataset folder (inside the zip) | split file | split URL
DATASETS=(
  "nyuv2/test|nyuv2_test.txt|$SPLITS/nyu_normals/nyuv2_test.txt"
  "ibims/ibims|ibims_test.txt|$SPLITS/ibims_normals/ibims_test.txt"
  "diode/val|diode_test.txt|$SPLITS/diode_normals/diode_test.txt"
)

command -v unzip >/dev/null || { echo "unzip is required (apt-get install -y unzip)" >&2; exit 1; }
mkdir -p "$TARGET"

verify() {  # every listed RGB + normal file exists
  local dir=$1 split=$2 missing
  [ -f "$TARGET/$dir/$split" ] || return 1
  missing=$(awk -v root="$TARGET/$dir" 'NF>=2 { print root "/" $1; print root "/" $2 }' "$TARGET/$dir/$split" |
            while read -r f; do [ -f "$f" ] || echo "$f"; done | head -n 5)
  [ -z "$missing" ] || { echo "$missing"; return 1; }
}

need_zip=0
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r dir split _ <<< "$entry"
  verify "$dir" "$split" >/dev/null || need_zip=1
done

if [ "$need_zip" = 1 ]; then
  echo "Downloading Marigold normals evaluation set (~21 GB, resumable) ..."
  curl -fL --retry 5 -C - -o "$ZIP" "$ZIP_URL"
  echo "Extracting NYUv2 / iBims-1 / DIODE (other datasets in the zip are skipped) ..."
  unzip -q -o "$ZIP" 'nyuv2/test/*' 'ibims/ibims/*' 'diode/val/*' -d "$TARGET"
  [ "${KEEP_ZIP:-0}" = 1 ] || rm -f "$ZIP"
fi

status=0
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r dir split url <<< "$entry"
  curl -fsSL --retry 5 -o "$TARGET/$dir/$split" "$url"
  if missing=$(verify "$dir" "$split"); then
    echo "OK  $dir: $(grep -c . "$TARGET/$dir/$split") samples listed in $TARGET/$dir/$split"
  else
    echo "MISSING files under $TARGET/$dir, e.g.:" >&2; echo "$missing" >&2; status=1
  fi
done
indoors=$(grep -c '^indoors/' "$TARGET/diode/val/diode_test.txt" || true)
echo "    DIODE: $indoors indoors (test) + $(grep -c '^outdoor/' "$TARGET/diode/val/diode_test.txt" || true) outdoor (calibration)"
exit $status
