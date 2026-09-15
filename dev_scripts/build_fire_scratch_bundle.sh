#!/usr/bin/env bash
# build_fire_scratch_bundle.sh - regenerate the two Colab upload artifacts for the scratch
# fire-model run and place them in their expected upload locations:
#
#   <out>/fire-scratch-train-colab.ipynb   (import into Colab)
#   <out>/<version>/scripts.zip            (upload to Drive: mazr3a-fire-scratch/<version>/)
#
# Unlike pack_fire_scratch_colab.sh, this does NOT rebuild the large data zips - it only
# rebuilds the notebook + the per-version scripts bundle (the parts that actually change).
#
# Usage:
#   ./dev_scripts/build_fire_scratch_bundle.sh [--out DIR] [--version v3]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="model-training/scratch-model/scratch-v3"
VERSION="v3"
while [ $# -gt 0 ]; do
  case "$1" in
    -o|--out) OUT_DIR="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# make the output dir absolute (the zip runs inside a subshell that cd's into $WORK)
case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac

NOTEBOOK="$OUT_DIR/fire-scratch-train-colab.ipynb"
SCRIPTS_ZIP="$OUT_DIR/$VERSION/scripts.zip"

echo "==> [1/3] build notebook -> $NOTEBOOK"
python3 dev_scripts/build_fire_scratch_colab_nb.py --out "$NOTEBOOK"

echo "==> [2/3] bundle scripts -> $SCRIPTS_ZIP"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/scripts"
cp dev_scripts/prep_fire_scratch_dataset.py \
   dev_scripts/prep_fireviewer_dataset.py \
   dev_scripts/dedup_fire_scratch.py \
   dev_scripts/augment_fire_train.py \
   dev_scripts/colab_train_scratch.py "$WORK/scripts/"
printf 'Fire scratch Colab upload (split bundles)\ncreated (UTC): %s\ngit: %s\nversion: %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse --short HEAD 2>/dev/null || echo n/a)" \
  "$VERSION" > "$WORK/BUNDLE.txt"
mkdir -p "$(dirname "$SCRIPTS_ZIP")"
( cd "$WORK" && zip -qr "$SCRIPTS_ZIP" scripts BUNDLE.txt )

echo "==> [3/3] verify expected locations"
[ -f "$NOTEBOOK" ]   || { echo "MISSING notebook: $NOTEBOOK" >&2; exit 1; }
[ -f "$SCRIPTS_ZIP" ] || { echo "MISSING scripts.zip: $SCRIPTS_ZIP" >&2; exit 1; }

echo
echo "notebook    : $NOTEBOOK   ($(du -h "$NOTEBOOK" | cut -f1))"
echo "scripts.zip : $SCRIPTS_ZIP   ($(du -h "$SCRIPTS_ZIP" | cut -f1))"
echo
echo "Upload (Google Drive):"
echo "  1. import  $NOTEBOOK"
echo "  2. upload  $SCRIPTS_ZIP  ->  MyDrive/mazr3a-fire-scratch/$VERSION/scripts.zip"
