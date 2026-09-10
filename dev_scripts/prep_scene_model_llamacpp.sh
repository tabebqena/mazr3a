#!/usr/bin/env bash
# ============================================================
# prep_scene_model_llamacpp.sh - fetch the SMALL scene-caption model onto the
# host, and optionally the llama.cpp binaries that serve it.
#
# ADD-ONLY BY DESIGN. This script NEVER deletes and NEVER re-downloads anything
# that already exists (use --force to replace a specific file), and it NEVER
# touches the RETAINED OpenVINO Qwen2-VL-2B export in models/scene/ - that model
# is kept deliberately so MODEL_BACKEND=openvino can be flipped back on without a
# download (see plans/event-scene-reader.md and models/scene/README.md).
#
# It is a plain `curl` downloader: no optimum-cli, no torch, no pip on the host
# (the old dev_scripts/prep_scene_model.sh needed curl+python3 and this keeps
# that promise).
#
# WHAT IT FETCHES
#   1) a small GGUF VLM + its mmproj vision projector, into its OWN subdir:
#        models/scene/smolvlm2-500m/<model>.gguf
#        models/scene/smolvlm2-500m/mmproj-<...>.gguf
#   2) optionally the llama.cpp server/CLI binaries into models/scene/bin/
#
# Usage:
#   bash dev_scripts/prep_scene_model_llamacpp.sh              # GGUF + mmproj
#   bash dev_scripts/prep_scene_model_llamacpp.sh --list       # show the repo files
#   bash dev_scripts/prep_scene_model_llamacpp.sh --force      # re-download
#   bash dev_scripts/prep_scene_model_llamacpp.sh --bin        # ALSO fetch llama.cpp
#   bash dev_scripts/prep_scene_model_llamacpp.sh --repo USER/MODEL
#   bash dev_scripts/prep_scene_model_llamacpp.sh --bin --llamacpp-tag b10900
#     (the tag is only needed for --bin; the script DISCOVERS the exact asset
#      name from the release, because llama.cpp ships .tar.gz now and the names
#      change over time. b10900 was verified 2026-09-10.)
#
# After running, point config/scenereader.conf at the two files it reports and
# `docker compose restart scenereader`.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models/scene/smolvlm2-500m"
BIN_DIR="${ROOT_DIR}/models/scene/bin"

REPO="ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
FORCE=""
DO_BIN=""
LLAMACPP_TAG=""
MODE="fetch"

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="${2:?--repo needs a value}"; shift 2 ;;
    --dest) DEST_DIR="${2:?--dest needs a value}"; shift 2 ;;
    --bin) DO_BIN="1"; shift ;;
    --llamacpp-tag) LLAMACPP_TAG="${2:?--llamacpp-tag needs a value}"; shift 2 ;;
    --force) FORCE="1"; shift ;;
    --list) MODE="list"; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found" >&2; exit 1; }

API="https://huggingface.co/api/models/${REPO}/tree/main"
BASE="https://huggingface.co/${REPO}/resolve/main"

echo "=============================================================="
echo "1) list ${REPO}"

# List the repo files once; pick the two we need by PATTERN rather than a
# hard-coded filename, so a repo rename of a quant does not break the script.
listing="$(curl -sL --fail --max-time 60 "$API")"
[ -n "$listing" ] || { echo "ERROR: could not list ${REPO} - wrong repo id?" >&2; exit 1; }

if [ "$MODE" = "list" ]; then
  echo "$listing" | python3 -c 'import sys,json
d=json.load(sys.stdin)
for x in d:
    if x.get("type")=="file":
        print("  {:>12}  {}".format(x.get("size",0), x["path"]))'
  exit 0
fi

picked="$(echo "$listing" | python3 -c 'import sys,json,re
d=json.load(sys.stdin)
files=[x["path"] for x in d if x.get("type")=="file" and x["path"].lower().endswith(".gguf")]
def pick(pats, exclude=()):
    for pat in pats:
        for f in files:
            if re.search(pat, f, re.I) and not any(re.search(e, f, re.I) for e in exclude):
                return f
    return ""
mmproj = pick([r"mmproj.*f16", r"mmproj.*f32", r"mmproj"])
model  = pick([r"q8_0", r"q6_k", r"q4_k_m", r"q4_k", r".*"], exclude=(r"mmproj",))
allf   = [x["path"] for x in d if x.get("type")=="file"]
imgs   = [f for f in allf if re.search(r"\.(png|jpe?g|webp|gif|mp4|mov)$", f, re.I)]
print("\n".join([mmproj, model, "|".join(files), "|".join(imgs)]))')"

MMPROJ="$(echo "$picked" | sed -n 1p)"
MODEL="$(echo "$picked" | sed -n 2p)"
GGUFS="$(echo "$picked" | sed -n 3p)"
VIDEOS="$(echo "$picked" | sed -n 4p)"

if [ -z "$MODEL" ] || [ -z "$MMPROJ" ]; then
  echo "ERROR: could not find a GGUF model + mmproj in ${REPO}." >&2
  echo "       available: ${GGUFS:-<none>}" >&2
  echo "       Use --repo to point at another GGUF repo, or --list to inspect it." >&2
  exit 1
fi

echo "   model  : ${MODEL}"
echo "   mmproj : ${MMPROJ}"

# ---------------------------------------------------------------
# 2) download only what is MISSING (add-only; --force replaces)
# ---------------------------------------------------------------
mkdir -p "$DEST_DIR"
fetch() {
  local path="$1"
  local target="$DEST_DIR/$(basename "$path")"
  if [ -s "$target" ] && [ -z "$FORCE" ]; then
    echo "   KEEP   $(basename "$path") (already present; --force to replace)"
    return 0
  fi
  echo "   GET    ${path}"
  curl -L --fail --retry 3 --max-time 7200 -# -o "$target" "${BASE}/${path}"
}

echo "=============================================================="
echo "2) download into ${DEST_DIR} (existing files are kept)"
fetch "$MODEL"
fetch "$MMPROJ"

# The RETAINED OpenVINO IR (if present) is never touched - make that explicit.
if compgen -G "${ROOT_DIR}/models/scene/openvino_*.xml" >/dev/null 2>&1; then
  echo "   NOTE   the retained OpenVINO IR in models/scene/ was left untouched"
fi

# ---------------------------------------------------------------
# 3) OPTIONAL: llama.cpp binaries (llama-server + llama-mtmd-cli)
# ---------------------------------------------------------------
if [ -n "$DO_BIN" ]; then
  echo "=============================================================="
  echo "3) llama.cpp binaries -> ${BIN_DIR}"
  if [ -z "$LLAMACPP_TAG" ]; then
    echo "ERROR: --bin needs --llamacpp-tag <release tag> so the download is" >&2
    echo "       reproducible (see https://github.com/ggml-org/llama.cpp/releases)." >&2
    echo "       Example: --bin --llamacpp-tag b10900  (verified 2026-09-10)" >&2
    echo "       The exact asset name is DISCOVERED from the release, so the tag is" >&2
    echo "       all that is needed - llama.cpp ships .tar.gz and renames assets." >&2
    exit 1
  fi
  mkdir -p "$BIN_DIR"
  # DISCOVER the asset name from the release instead of assuming it - llama.cpp
  # asset names have changed over time and assuming one breaks the fetch with a
  # confusing 404. Prefer a plain Ubuntu x64 CPU build (never a CUDA/Vulkan/ROCm/
  # SYCL/ARM/macOS/Windows asset).
  REL_API="https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${LLAMACPP_TAG}"
  ASSET="$(curl -sL --fail --max-time 60 "$REL_API" | python3 -c '
import sys, json, re
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
names = [a["name"] for a in d.get("assets", [])]
def bad(n):
    # never a GPU/accelerator or foreign-platform build; "openvino" is skipped too
    # so the PLAIN CPU build (the smallest, ~17 MB) wins by default
    return re.search(r"cuda|vulkan|rocm|sycl|hip|openvino|arm|aarch|s390|riscv|android|macos|win|ios|xcframework", n, re.I)
def pkg(n):
    # llama.cpp ships .tar.gz now (it used .zip historically) - accept both
    return n.endswith(".tar.gz") or n.endswith(".tgz") or n.endswith(".zip")
cands = [n for n in names if pkg(n) and not bad(n) and re.search(r"ubuntu", n, re.I) and re.search(r"x64|x86_64", n, re.I)]
if not cands:
    cands = [n for n in names if pkg(n) and not bad(n)]
print(cands[0] if cands else "")')"
  if [ -z "$ASSET" ]; then
    echo "ERROR: no suitable CPU asset found in release ${LLAMACPP_TAG}." >&2
    echo "       Available assets:" >&2
    curl -sL --fail --max-time 60 "$REL_API" | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
for a in d.get("assets", []):
    print("         " + a["name"])' >&2 || true
    echo "       Pick another --llamacpp-tag (see the llama.cpp releases page)." >&2
    exit 1
  fi
  echo "   asset  : ${ASSET}"
  URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMACPP_TAG}/${ASSET}"
  ZIP="${BIN_DIR}/${ASSET}"
  if [ -s "$ZIP" ] && [ -z "$FORCE" ]; then
    echo "   KEEP   ${ASSET}"
  else
    echo "   GET    ${ASSET}"
    curl -L --fail --retry 3 --max-time 3600 -# -o "$ZIP" "$URL"
  fi
  tmp="$(mktemp -d)"
  case "$ASSET" in
    *.tar.gz|*.tgz)
      command -v tar >/dev/null 2>&1 || { echo "ERROR: tar not found" >&2; exit 1; }
      tar -xzf "$ZIP" -C "$tmp" ;;
    *.zip)
      command -v unzip >/dev/null 2>&1 || { echo "ERROR: unzip not found" >&2; exit 1; }
      unzip -q -o "$ZIP" -d "$tmp" ;;
    *) echo "ERROR: unsupported archive ${ASSET}" >&2; exit 1 ;;
  esac
  # flatten: the archive nests the binaries in a subdir; copy the two we run
  found=0
  for want in llama-server llama-mtmd-cli; do
    src="$(find "$tmp" -type f -name "$want" | head -n1)"
    if [ -n "$src" ]; then
      cp -f "$src" "${BIN_DIR}/${want}"
      chmod 755 "${BIN_DIR}/${want}"
      echo "   OK     ${BIN_DIR}/${want}"
      found=1
    else
      echo "   MISS   ${want} (is this the plain CPU build?)" >&2
    fi
  done
  rm -rf "$tmp"
  [ "$found" = "1" ] || { echo "ERROR: neither binary found in ${ASSET}" >&2; exit 1; }
fi

# ---------------------------------------------------------------
# 4) report the config values to paste
# ---------------------------------------------------------------
echo "=============================================================="
echo "4) point config/scenereader.conf at:"
echo "   MODEL_FILE=/models/scene/$(basename "$DEST_DIR")/$(basename "$MODEL")"
echo "   MMPROJ_FILE=/models/scene/$(basename "$DEST_DIR")/$(basename "$MMPROJ")"
if [ -n "$DO_BIN" ]; then
  echo "   LLAMA_SERVER_BIN=/models/scene/bin/llama-server"
  echo "   LLAMA_CLI_BIN=/models/scene/bin/llama-mtmd-cli"
else
  echo "   (run again with --bin --llamacpp-tag <tag> to fetch llama.cpp itself)"
fi
echo "   then: docker compose restart scenereader"
