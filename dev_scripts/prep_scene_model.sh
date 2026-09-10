#!/usr/bin/env bash
# ============================================================
# Prepare the scenewatch scene-description model into the OpenVINO IR
# directory scenewatch expects at models/scene/.
#
# DEFAULT (no deps): DOWNLOAD a pre-converted SmolVLM2 OpenVINO export from
# the Hugging Face Hub with plain `curl`. No optimum-cli, no torch, no pip -
# which is what a constrained/small host needs.
#
# This is a GIT-IGNORED artifact (models/scene/ is not tracked - see
# models/scene/README.md and .gitignore); it does NOT ride git.
#
# Usage:
#   ./dev_scripts/prep_scene_model.sh              # default: int4 export (~356 MB)
#   ./dev_scripts/prep_scene_model.sh --repo int8  # 8-bit export (~509 MB)
#   ./dev_scripts/prep_scene_model.sh --repo fp16  # full precision (~2.0 GB)
#   ./dev_scripts/prep_scene_model.sh --repo 256m  # 256M fp16 (~1.0 GB)
#   ./dev_scripts/prep_scene_model.sh --repo <user/model>   # any pre-converted OV repo
#   ./dev_scripts/prep_scene_model.sh --list       # show the curated repos
#   ./dev_scripts/prep_scene_model.sh --export     # BUILD locally with optimum-cli
#                                                  #   (needs `optimum[openvino]`, pulls torch)
#
# Requirements: `curl` + `python3` (both already used elsewhere in this repo).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models/scene"

MODE="download"
REPO="int4"
WEIGHT_FORMAT="int4"
TRUST_REMOTE=""
FORCE=""

# Curated pre-converted OpenVINO exports (verified to exist + their sizes).
#   int4   smallest AND ships explicit OpenVINO tokenizer/detokenizer IRs
#          (widest openvino-genai compatibility)          ~356 MB
#   int8   8-bit weight-only export from an HF/optimum
#          maintainer (trusted provenance)                ~509 MB
#   fp16   full precision from the same trusted source     ~2.0 GB
#   256m   smaller 500M model at full precision            ~1.0 GB
alias_for() {
  case "$1" in
    int4)  echo "circulus/SmolVLM2-500M-ov-sym-int4" ;;
    int8)  echo "echarlaix/SmolVLM2-500M-Video-Instruct-openvino-8bit-woq" ;;
    fp16)  echo "echarlaix/SmolVLM2-500M-Video-Instruct-openvino" ;;
    256m)  echo "echarlaix/SmolVLM2-256M-Video-Instruct-openvino" ;;
    *)     echo "$1" ;;   # already a "user/model" repo id
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="${2:?--repo needs a value}"; shift 2 ;;
    --dest) DEST_DIR="${2:?--dest needs a value}"; shift 2 ;;
    --export) MODE="export"; shift ;;
    --weight-format) WEIGHT_FORMAT="${2:?--weight-format needs a value}"; shift 2 ;;
    --trust-remote-code) TRUST_REMOTE="--trust-remote-code"; shift ;;
    --force) FORCE="1"; shift ;;
    --list)
      echo "curated pre-converted OpenVINO repos:"
      echo "  int4 -> $(alias_for int4)   (~356 MB, complete OV tokenizer/detokenizer IRs)"
      echo "  int8 -> $(alias_for int8)   (~509 MB, trusted source)"
      echo "  fp16 -> $(alias_for fp16)   (~2.0 GB)"
      echo "  256m -> $(alias_for 256m)   (~1.0 GB)"
      exit 0 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

MODEL_ID="$(alias_for "$REPO")"

# Refuse to clobber an existing export unless --force (README/VERSIONS.md are
# ours and always kept).
if [ -d "$DEST_DIR" ] && [ -z "$FORCE" ]; then
  if find "$DEST_DIR" -maxdepth 1 -type f \
       ! -name 'README.md' ! -name 'VERSIONS.md' | grep -q .; then
    echo "ERROR: ${DEST_DIR} already holds a model." >&2
    echo "       Re-run with --force to replace it, or --dest DIR to export elsewhere." >&2
    exit 1
  fi
fi

mkdir -p "$DEST_DIR"

if [ "$MODE" = "export" ]; then
  # ==========================================================
  # BUILD PATH (dev machine): optimum-cli export openvino
  # ==========================================================
  if ! command -v optimum-cli >/dev/null 2>&1; then
    echo "ERROR: optimum-cli not found." >&2
    echo "       pip install -U 'optimum[openvino]' openvino   # note: pulls torch (~2 GB)" >&2
    echo "       or drop --export to DOWNLOAD a pre-converted export instead (no deps)." >&2
    exit 1
  fi
  echo "=============================================================="
  echo "1) export ${MODEL_ID} -> OpenVINO IR (${WEIGHT_FORMAT})"
  echo "   dest: ${DEST_DIR}"
  EXPORT_LOG="$(mktemp)"
  set +e
  optimum-cli export openvino --model "$MODEL_ID" --weight-format "$WEIGHT_FORMAT" \
    --task image-text-to-text $TRUST_REMOTE "$DEST_DIR" >"$EXPORT_LOG" 2>&1
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then
    echo "   (export with --task failed, retrying without it)"
    tail -n 5 "$EXPORT_LOG" || true
    optimum-cli export openvino --model "$MODEL_ID" --weight-format "$WEIGHT_FORMAT" \
      $TRUST_REMOTE "$DEST_DIR"
  fi
  rm -f "$EXPORT_LOG"
else
  # ==========================================================
  # DOWNLOAD PATH (default): pre-converted OpenVINO export, curl only
  # ==========================================================
  command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found" >&2; exit 1; }
  command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 1; }

  API="https://huggingface.co/api/models/${MODEL_ID}/tree/main"
  BASE="https://huggingface.co/${MODEL_ID}/resolve/main"

  echo "=============================================================="
  echo "1) download ${MODEL_ID} (pre-converted OpenVINO export)"
  echo "   dest: ${DEST_DIR}"
  echo "   (no optimum-cli / torch needed - plain curl)"

  paths="$(curl -sL --fail --max-time 60 "$API" \
    | python3 -c 'import sys,json
d=json.load(sys.stdin)
for x in d:
    if x.get("type")=="file" and x["path"] not in (".gitattributes","README.md"):
        print(x["path"], x.get("size",0))')"
  if [ -z "$paths" ]; then
    echo "ERROR: no files listed for ${MODEL_ID} - wrong repo id?" >&2
    exit 1
  fi

  total=0
  while read -r path size; do
    [ -n "$path" ] || continue
    mkdir -p "$DEST_DIR/$(dirname "$path")"
    echo "   - ${path} (${size} bytes)"
    curl -L --fail --retry 3 --max-time 3600 -# -o "$DEST_DIR/${path}" "$BASE/${path}"
    [ "$size" = "0" ] || total=$((total + size))
  done <<< "$paths"

  echo "   downloaded ~$((total / 1024 / 1024)) MB"
fi

# ============================================================
# 2) compatibility check (what openvino_genai.VLMPipeline needs)
# ============================================================
echo "=============================================================="
echo "2) verify the export layout"
missing=0
for f in config.json openvino_language_model.xml openvino_vision_embeddings_model.xml \
         openvino_text_embeddings_model.xml; do
  if [ -f "${DEST_DIR}/${f}" ]; then echo "   OK   ${f}"; else echo "   MISS ${f}"; missing=1; fi
done
# tokenizer / detokenizer: explicit OpenVINO IRs OR the HF tokenizer.json
if [ -f "${DEST_DIR}/openvino_tokenizer.xml" ]; then
  echo "   OK   openvino_tokenizer.xml"
elif [ -f "${DEST_DIR}/tokenizer.json" ]; then
  echo "   OK   tokenizer.json (no OV tokenizer IR; needs a recent openvino-genai)"
else
  echo "   MISS tokenizer (openvino_tokenizer.xml or tokenizer.json)"; missing=1
fi
if [ -f "${DEST_DIR}/openvino_detokenizer.xml" ]; then
  echo "   OK   openvino_detokenizer.xml"
elif [ -f "${DEST_DIR}/tokenizer.json" ]; then
  echo "   WARN no openvino_detokenizer.xml - openvino-genai must build it from tokenizer.json."
  echo "        If scenewatch --check fails to load, re-run with: $0 --repo int4"
else
  echo "   MISS detokenizer"; missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "ERROR: the export is incomplete - scenewatch will not load it." >&2
  exit 1
fi

biggest="$(ls -S "${DEST_DIR}"/*.bin 2>/dev/null | head -1 || true)"
if [ -n "$biggest" ]; then
  echo "   md5 $(md5sum "$biggest" | awk '{print $1}')  $(basename "$biggest")"
fi
echo "   on-disk: $(du -sh "$DEST_DIR" | awk '{print $1}')"

# ============================================================
# 3) provenance line for models/scene/VERSIONS.md
# ============================================================
echo "=============================================================="
echo "3) append to models/scene/VERSIONS.md:"
echo "   | $(date -u +%Y-%m-%d) | ${MODEL_ID} | ${MODE} | $(du -sh "$DEST_DIR" | awk '{print $1}') | $( [ -n "$biggest" ] && md5sum "$biggest" | awk '{print $1}' || echo '-') | ACTIVE |"
echo
echo "DONE."
echo "IMPORTANT:"
echo "  - models/scene/ is GIT-IGNORED. Do NOT 'git add' it."
echo "  - config/scenewatch.conf MODEL_DIR is the CONTAINER path /models/scene"
echo "    (the compose service mounts ./models at /models:ro)."
echo "  - Verify the load + one caption:"
echo "      docker compose exec scenewatch python /scenewatch/scenewatch.py --check"
