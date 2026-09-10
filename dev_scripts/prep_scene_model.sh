#!/usr/bin/env bash
# ============================================================
# Prepare the RETAINED OpenVINO VLM for the scenereader service into
# models/scene/ - the directory the `openvino` captioner backend reads.
#
# THIS IS THE LARGER, FALLBACK BACKEND. The DEFAULT (small) model is a GGUF
# fetched by dev_scripts/prep_scene_model_llamacpp.sh and run by llama.cpp; this
# script exists because the already-downloaded IR is KEPT so the operator can
# reassess a bigger model later WITHOUT re-downloading it. It is ADD-ONLY: an
# existing export is never replaced unless --force is given, so it can never
# silently clobber what is on the host, and it never touches the small GGUF or
# models/scene/bin/.
#
# DEFAULT (no deps): DOWNLOAD a pre-converted VLM OpenVINO export from the
# Hugging Face Hub with plain `curl`. No optimum-cli, no torch, no pip.
#
# WHICH MODEL, AND WHY: openvino_genai.VLMPipeline implements a CLOSED list of
# VLM architectures - llava, qwen2_vl, qwen2_5_vl, gemma3, minicpm, phi3_v,
# phi4mm (verified by inspecting libopenvino_genai.so). SmolVLM is NOT on that
# list, and OpenVINO's own org only publishes 7B variants (far too heavy for
# this host), so the default here is **Qwen2-VL-2B-Instruct int4** - the
# smallest VLM the runtime can actually load (~1.76 GB).
#
# This is a GIT-IGNORED artifact (models/scene/ is not tracked - see
# models/scene/README.md and .gitignore); it does NOT ride git.
#
# Usage:
#   ./dev_scripts/prep_scene_model.sh              # default: Qwen2-VL-2B int4 (~1.76 GB)
#   ./dev_scripts/prep_scene_model.sh --list       # show the curated repos
#   ./dev_scripts/prep_scene_model.sh --repo 2b25  # Qwen2.5-VL-3B int4 (~2.5 GB)
#   ./dev_scripts/prep_scene_model.sh --repo <user/model>   # any pre-converted OV VLM repo
#   ./dev_scripts/prep_scene_model.sh --export --model <hf-id>   # BUILD locally instead
#                                                  # (needs `optimum[openvino]`, pulls torch)
#
# Requirements: `curl` + `python3` (both already used elsewhere in this repo).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models/scene"

MODE="download"
REPO="2b"
MODEL_ID=""
WEIGHT_FORMAT="int4"
TRUST_REMOTE=""
FORCE=""

# Curated pre-converted OpenVINO VLM exports (verified to exist + sizes).
# Every one of these is an architecture the runtime implements.
#   2b   Qwen2-VL-2B   int4  ~1.76 GB  <-- default (smallest supported)
#   2b25 Qwen2.5-VL-3B int4  ~2.5 GB   (newer family, more RAM)
#   7b   Qwen2-VL-7B   int4  ~5 GB     (documented for completeness - NOT
#                                       for this 7.5 GB host)
alias_for() {
  case "$1" in
    2b)   echo "helenai/Qwen2-VL-2B-Instruct-ov-int4" ;;
    2b25) echo "llmware/Qwen2.5-VL-3B-Instruct-ov-int4" ;;
    7b)   echo "OpenVINO/Qwen2-VL-7B-Instruct-int4-ov" ;;
    *)    echo "$1" ;;   # already a "user/model" repo id
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="${2:?--repo needs a value}"; shift 2 ;;
    --model) MODEL_ID="${2:?--model needs a value}"; shift 2 ;;
    --dest) DEST_DIR="${2:?--dest needs a value}"; shift 2 ;;
    --export) MODE="export"; shift ;;
    --weight-format) WEIGHT_FORMAT="${2:?--weight-format needs a value}"; shift 2 ;;
    --trust-remote-code) TRUST_REMOTE="--trust-remote-code"; shift ;;
    --force) FORCE="1"; shift ;;
    --list)
      echo "curated pre-converted OpenVINO VLM repos:"
      echo "  2b   -> $(alias_for 2b)      (~1.76 GB, int4)  [default]"
      echo "  2b25 -> $(alias_for 2b25)  (~2.5 GB,  int4)"
      echo "  7b   -> $(alias_for 7b)  (~5 GB,   int4, too big for this host)"
      exit 0 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

[ -n "$MODEL_ID" ] || MODEL_ID="$(alias_for "$REPO")"

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
  echo "1) download ${MODEL_ID} (pre-converted OpenVINO VLM export)"
  echo "   dest: ${DEST_DIR}"
  echo "   (no optimum-cli / torch needed - plain curl; this is a large download)"

  paths="$(curl -sL --fail --max-time 60 "$API" \
    | python3 -c 'import sys,json
d=json.load(sys.stdin)
if isinstance(d,dict):
    sys.exit(1)
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
    curl -L --fail --retry 3 --max-time 7200 -# -o "$DEST_DIR/${path}" "$BASE/${path}"
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
for f in config.json openvino_language_model.xml \
         openvino_vision_embeddings_model.xml; do
  if [ -f "${DEST_DIR}/${f}" ]; then echo "   OK   ${f}"; else echo "   MISS ${f}"; missing=1; fi
done
# Text embeddings OR the vision-embeddings MERGER (Qwen2-VL ships the merger).
if [ -f "${DEST_DIR}/openvino_text_embeddings_model.xml" ]; then
  echo "   OK   openvino_text_embeddings_model.xml"
elif [ -f "${DEST_DIR}/openvino_vision_embeddings_merger_model.xml" ]; then
  echo "   OK   openvino_vision_embeddings_merger_model.xml"
else
  echo "   MISS text embeddings / vision-embeddings merger"; missing=1
fi
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
else
  echo "   MISS detokenizer"; missing=1
fi
if [ "$missing" -ne 0 ]; then
  echo "ERROR: the export is incomplete - scenewatch will not load it." >&2
  exit 1
fi

# The declared architecture MUST be one the runtime implements, or
# VLMPipeline fails at load with "Unsupported '<type>' VLM model type".
MT="$(python3 - "${DEST_DIR}/config.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        print(json.load(fh).get("model_type", ""))
except Exception:
    print("")
PY
)"
echo "   model_type: ${MT:-<unknown>}"
case "$MT" in
  qwen2_vl|qwen2_5_vl|llava|gemma3|minicpm|phi3_v|phi4mm)
    echo "   OK   architecture is supported by openvino-genai" ;;
  smolvlm|smolvlm2|idefics3|"")
    echo "ERROR: '${MT:-unknown}' is NOT supported by openvino-genai's VLM loader." >&2
    echo "       Supported: llava, qwen2_vl, qwen2_5_vl, gemma3, minicpm, phi3_v, phi4mm." >&2
    exit 1 ;;
  *)
    echo "   WARN '${MT}' is not in the known-supported list - if the container logs" >&2
    echo "        \"Unsupported '<type>' VLM model type\", pick another --repo." >&2 ;;
esac

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
echo "  - Verify the load + one caption (start scenewatch first):"
echo "      docker compose exec scenewatch python /scenewatch/scenewatch.py --check"
