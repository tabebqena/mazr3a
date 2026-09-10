#!/usr/bin/env bash
# ============================================================
# Prepare the scenewatch scene-description model into the OpenVINO IR
# directory scenewatch expects at models/scene/.
#
# This is a GIT-IGNORED, ~500 MB artifact (models/scene/ is not tracked -
# see models/scene/README.md and .gitignore); it is produced here and
# fetched onto the host as a deploy prerequisite, it does NOT ride git.
#
# Works on any machine with python + optimum-intel + openvino installed
# (a dev box, or the host itself if it has the disk + network).
#
# Usage:
#   ./dev_scripts/prep_scene_model.sh [--model 500m|256m|<hf-id>]
#                                     [--weight-format int4|int8|fp16]
#                                     [--dest DIR] [--trust-remote-code]
#                                     [--force]
# Examples:
#   ./dev_scripts/prep_scene_model.sh                 # SmolVLM2-500M INT4 (default)
#   ./dev_scripts/prep_scene_model.sh --model 256m    # RAM fallback
#   ./dev_scripts/prep_scene_model.sh --weight-format int8
#
# Requires: pip install -U "optimum[openvino]" openvino  (provides optimum-cli)
# The exported dir is self-contained: config.json + processor/tokenizer
# assets + openvino_model.xml/.bin, which is exactly what
# openvino_genai.VLMPipeline(<MODEL_DIR>) loads.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models/scene"

MODEL="500m"
WEIGHT_FORMAT="int4"
TRUST_REMOTE=""
FORCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:?--model needs a value}"; shift 2 ;;
    --weight-format) WEIGHT_FORMAT="${2:?--weight-format needs a value}"; shift 2 ;;
    --dest) DEST_DIR="${2:?--dest needs a value}"; shift 2 ;;
    --trust-remote-code) TRUST_REMOTE="--trust-remote-code"; shift ;;
    --force) FORCE="1"; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Resolve the shorthand to a Hugging Face repo id.
case "$MODEL" in
  500m|smolvlm2-500m) MODEL_ID="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" ;;
  256m|smolvlm2-256m) MODEL_ID="HuggingFaceTB/SmolVLM2-256M-Video-Instruct" ;;
  *) MODEL_ID="$MODEL" ;;   # already an HF repo id
esac

if ! command -v optimum-cli >/dev/null 2>&1; then
  echo "ERROR: optimum-cli not found - pip install -U 'optimum[openvino]' openvino" >&2
  exit 1
fi

if [ -d "$DEST_DIR" ] && [ -n "$(ls -A "$DEST_DIR" 2>/dev/null | grep -v '^README.md$' | grep -v '^VERSIONS.md$' || true)" ] && [ -z "$FORCE" ]; then
  echo "ERROR: ${DEST_DIR} already holds a model." >&2
  echo "       Re-run with --force to replace it, or --dest DIR to export elsewhere." >&2
  exit 1
fi

echo "=============================================================="
echo "1) export ${MODEL_ID} -> OpenVINO IR (${WEIGHT_FORMAT})"
echo "   dest: ${DEST_DIR}"
mkdir -p "$DEST_DIR"

# OpenVINO GenAI loads a VLMPipeline from this exact layout. `--task
# image-text-to-text` is passed explicitly so optimum does not have to infer
# it; if this optimum release rejects the flag we retry without it.
set +e
EXPORT_LOG="$(mktemp)"
optimum-cli export openvino \
  --model "$MODEL_ID" \
  --weight-format "$WEIGHT_FORMAT" \
  --task image-text-to-text \
  $TRUST_REMOTE \
  "$DEST_DIR" >"$EXPORT_LOG" 2>&1
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "   (export with --task failed, retrying without it)"
  tail -n 5 "$EXPORT_LOG" || true
  optimum-cli export openvino \
    --model "$MODEL_ID" \
    --weight-format "$WEIGHT_FORMAT" \
    $TRUST_REMOTE \
    "$DEST_DIR"
fi
rm -f "$EXPORT_LOG"

# Sanity: VLMPipeline (and scenewatch.py) needs both of these.
if [ ! -f "${DEST_DIR}/openvino_model.xml" ] || [ ! -f "${DEST_DIR}/config.json" ]; then
  echo "ERROR: export did not produce openvino_model.xml + config.json in ${DEST_DIR}" >&2
  ls -la "$DEST_DIR" >&2 || true
  exit 1
fi

echo "=============================================================="
echo "2) provenance (append this line to models/scene/VERSIONS.md)"
DATE_UTC="$(date -u +%Y-%m-%d)"
MD5="$(md5sum "${DEST_DIR}/openvino_model.bin" 2>/dev/null | awk '{print $1}')"
SIZE_H="$(du -sh "$DEST_DIR" | awk '{print $1}')"
echo "   | ${DATE_UTC} | ${MODEL_ID} | ${WEIGHT_FORMAT} | ${SIZE_H} | ${MD5} | ACTIVE |"

echo "=============================================================="
echo "3) DONE. Files in ${DEST_DIR}:"
ls -la "$DEST_DIR"
echo
echo "IMPORTANT:"
echo "  - models/scene/ is GIT-IGNORED (a ~500 MB binary must not ride git)."
echo "    Do NOT 'git add' it; fetch it onto the host as a deploy prerequisite."
echo "  - Point config/scenewatch.conf MODEL_DIR at /models/scene (the container"
echo "    path; the compose service mounts ./models at /models:ro)."
echo "  - Verify the load + one caption:"
echo "      docker compose exec scenewatch python /scenewatch/scenewatch.py --check"
