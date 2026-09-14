#!/usr/bin/env bash
# pack_fire_scratch_colab.sh - build the SPLIT Colab upload bundles for the scratch fire-model run.
#
# Data folders are zipped INDIVIDUALLY so a re-run only re-uploads what actually changed:
#
#   <out>/uploads/negatives.zip
#   <out>/uploads/domain_test.zip
#   <out>/uploads/cctv_emergency.zip
#   <out>/uploads/salah_haismawi.zip
#
# Scripts change often, so they ship per-version (each run uploads its own):
#
#   <out>/<version>/scripts.zip
#
# UPLOAD (Google Drive):
#   the four data zips -> MyDrive/mazr3a-fire-scratch/uploads/
#   scripts.zip        -> MyDrive/mazr3a-fire-scratch/<version>/scripts.zip
#
# The notebook re-uses an already-uploaded data zip when its local folder already exists, and
# only downloads/extracts the ones it still needs (see build_fire_scratch_colab_nb.py Cell 4).
#
# Usage:
#   ./dev_scripts/pack_fire_scratch_colab.sh [--out DIR] [--version v3] [--negatives true|false]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MT="model-training"
OUT_DIR="$MT/runs/scratch-v3"
VERSION="v3"
NEGATIVES=true
while [ $# -gt 0 ]; do
  case "$1" in
    --negatives)
      case "${2:-}" in
        true|TRUE|1)   NEGATIVES=true ;;
        false|FALSE|0) NEGATIVES=false ;;
        *) echo "usage: --negatives true|false" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    -o|--out) OUT_DIR="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# make the output dir absolute: the zip commands run inside a subshell that cd's into $WORK
case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac

UPLOADS_DIR="$OUT_DIR/uploads"
SCRIPTS_DIR="$OUT_DIR/$VERSION"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/negatives" "$WORK/domain_test" "$WORK/scripts"
mkdir -p "$WORK/cctv_emergency"
mkdir -p "$WORK/salah_haismawi"

# Copy images only - RECURSIVELY (some sets nest under images/camNN/) - and never the
# annotated (overlay-burned) variants, which would teach the model the drawn boxes.
# A basename collision across subdirs is disambiguated with the parent dir name.
copy_imgs() {
  local src="$1" dst="$2" f b p
  [ -d "$src" ] || { echo "  (skip, absent) $src"; return 0; }
  while IFS= read -r -d '' f; do
    b="$(basename "$f")"
    if [ -e "$dst/$b" ]; then
      p="$(basename "$(dirname "$f")")"
      b="${p}__${b}"
    fi
    cp "$f" "$dst/$b"
  done < <(find "$src" -type f \
                \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
                ! -iname '*annotated*' -print0)
}

if [ "$NEGATIVES" = true ]; then
  echo "collecting domain negatives -> $WORK/negatives"
  copy_imgs "$MT/ours/negatives/default-other"                          "$WORK/negatives"
  copy_imgs "$MT/ours/negatives/climate"                                "$WORK/negatives"
  copy_imgs "$MT/ours/negatives/dogs"                                   "$WORK/negatives"
  copy_imgs "$MT/ours/negatives/places"                                 "$WORK/negatives"
  copy_imgs "$MT/ours/cctv/dog-fp-alerts"                                 "$WORK/negatives"
  copy_imgs "$MT/ours/camera-clips/fire_events_over_0_5_false_positives"  "$WORK/negatives"
else
  echo "skipping domain negatives (--negatives false)"
fi

echo "collecting our-domain CCTV true fires (held-out test) -> $WORK/domain_test"
copy_imgs "$MT/ours/camera-clips/fire_events_over_0_5"                  "$WORK/domain_test"

echo "bundling CCTV Emergency (flat images+labels YOLO layout, test-only)"
if [ -d "$MT/sources/cctv_emergency" ]; then
  cp -r "$MT/sources/cctv_emergency/." "$WORK/cctv_emergency/"
else
  echo "  (skip, absent) $MT/sources/cctv_emergency"
fi

echo "bundling SalahALHaismawi (Roboflow 'Fire Detection.v1i.yolov8' -> YOLO tree, train)"
SALAH_ZIP="$MT/sources/salah_haismawi/dataset/Fire Detection.v1i.yolov8.zip"
if [ -f "$SALAH_ZIP" ]; then
  unzip -q -o "$SALAH_ZIP" -d "$WORK/salah_haismawi"
else
  echo "  (skip, absent) $SALAH_ZIP"
fi

echo "bundling scripts"
cp dev_scripts/prep_fire_scratch_dataset.py \
   dev_scripts/prep_fireviewer_dataset.py \
   dev_scripts/dedup_fire_scratch.py \
   dev_scripts/augment_fire_train.py \
   dev_scripts/colab_train_scratch.py "$WORK/scripts/"

printf 'Fire scratch Colab upload (split bundles)\ncreated (UTC): %s\ngit: %s\nversion: %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse --short HEAD 2>/dev/null || echo n/a)" \
  "$VERSION" > "$WORK/BUNDLE.txt"

mkdir -p "$UPLOADS_DIR" "$SCRIPTS_DIR"

echo "writing split data bundles -> $UPLOADS_DIR"
for f in negatives domain_test cctv_emergency salah_haismawi; do
  if [ -d "$WORK/$f" ] && [ -n "$(find "$WORK/$f" -type f -print -quit 2>/dev/null)" ]; then
    ( cd "$WORK" && zip -qr "$UPLOADS_DIR/$f.zip" "$f" )
  else
    echo "  (skip empty) $f"
  fi
done

echo "writing per-version scripts bundle -> $SCRIPTS_DIR/scripts.zip"
( cd "$WORK" && zip -qr "$SCRIPTS_DIR/scripts.zip" scripts BUNDLE.txt )

echo
for f in negatives domain_test cctv_emergency salah_haismawi; do
  [ -f "$UPLOADS_DIR/$f.zip" ] && echo "  $f: $(du -h "$UPLOADS_DIR/$f.zip" | cut -f1)"
done
echo "  scripts: $(du -h "$SCRIPTS_DIR/scripts.zip" | cut -f1)"
echo
echo "Upload:"
echo "  the four data zips -> MyDrive/mazr3a-fire-scratch/uploads/"
echo "  scripts.zip       -> MyDrive/mazr3a-fire-scratch/$VERSION/scripts.zip"
echo "then run the notebook generated by dev_scripts/build_fire_scratch_colab_nb.py"
