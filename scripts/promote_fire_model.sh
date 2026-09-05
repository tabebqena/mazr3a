#!/usr/bin/env bash
# ============================================================
# Promote a versioned fire/smoke checkpoint to ACTIVE.
#
# Copies <versions>/<version>/model.pt -> models/fire/best.pt and, when the
# version dir already carries a ready OpenVINO IR (best.xml + best.bin +
# labelmap.txt), copies those to models/fire/ too (deploy-ready). Otherwise it
# prints the exact prep command to regenerate the IR and warns that the running
# container still uses the previous IR until then.
#
# Usage:
#   ./scripts/promote_fire_model.sh <version-dir|unique-prefix>
# Examples:
#   ./scripts/promote_fire_model.sh v2-2026-09-05-hf-abonia877-ft5ep
#   ./scripts/promote_fire_model.sh v2            # unique-prefix match
#
# Versioning convention + layout: see models/fire/VERSIONS.md and
# plans/model-versioning.md. After promoting, update the version's status in
# its VERSION.json + VERSIONS.md, then deploy & verify on the host
# (per .roo/rules/sshuser.md):
#   ./scripts/deploy_firewatch.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSIONS_DIR="${ROOT_DIR}/models/fire/versions"
DEST_DIR="${ROOT_DIR}/models/fire"

ARG="${1:-}"
if [ -z "$ARG" ]; then
  echo "usage: $(basename "$0") <version-dir|unique-prefix>" >&2
  echo "e.g.:  $(basename "$0") v2-2026-09-05-hf-abonia877-ft5ep" >&2
  exit 1
fi

if [ ! -d "$VERSIONS_DIR" ]; then
  echo "ERROR: no version store at ${VERSIONS_DIR}" >&2
  exit 1
fi

# Resolve: exact dir name, or a UNIQUE prefix of a directory name.
VERSION_DIR=""
if [ -d "${VERSIONS_DIR}/${ARG}" ]; then
  VERSION_DIR="${VERSIONS_DIR}/${ARG}"
else
  matches=( "$VERSIONS_DIR"/"$ARG"* )
  if [ "${#matches[@]}" -eq 1 ] && [ -d "${matches[0]}" ]; then
    VERSION_DIR="${matches[0]}"
  else
    echo "ERROR: no version matches '${ARG}' under ${VERSIONS_DIR}" >&2
    echo "Available versions:" >&2
    ls -1 "$VERSIONS_DIR" >&2
    exit 1
  fi
fi

VERSION_ID="$(basename "$VERSION_DIR")"
echo "promoting version: ${VERSION_ID}"

# --- locate the checkpoint (model.pt preferred, best.pt fallback) -----------
PT=""
for cand in "${VERSION_DIR}/model.pt" "${VERSION_DIR}/best.pt"; do
  if [ -f "$cand" ]; then PT="$cand"; break; fi
done
if [ -z "$PT" ]; then
  echo "ERROR: no model.pt/best.pt in ${VERSION_DIR}" >&2
  exit 1
fi

echo "=============================================================="
echo "1) copy checkpoint -> ${DEST_DIR}/best.pt"
cp "$PT" "${DEST_DIR}/best.pt"
echo "   $(basename "$PT") ($(du -h "$PT" | cut -f1)) -> best.pt"

echo "=============================================================="
echo "2) OpenVINO IR bundled in this version?"
IR_READY=1
for f in best.xml best.bin labelmap.txt; do
  [ -f "${VERSION_DIR}/${f}" ] || IR_READY=0
done
if [ "$IR_READY" -eq 1 ]; then
  for f in best.xml best.bin labelmap.txt; do
    cp "${VERSION_DIR}/${f}" "${DEST_DIR}/${f}"
  done
  echo "   copied best.xml + best.bin + labelmap.txt -> ${DEST_DIR} (deploy-ready)"
else
  echo "   no best.xml/best.bin/labelmap.txt in this version."
  echo "   The ACTIVE .pt is updated, but the RUNNING container still uses the"
  echo "   previous OpenVINO IR until you regenerate + deploy it. Generate IR from"
  echo "   the new best.pt (class order is fire/other/smoke for both current versions;"
  echo "   labelmap.txt MUST match the checkpoint's class index order):"
  echo "     ./scripts/prep_fire_model.sh models/fire/best.pt 640 \"fire,other,smoke\""
fi

echo "=============================================================="
echo "3) next steps"
echo "   - record the promotion: set \"status\" in ${VERSION_DIR}/VERSION.json and"
echo "     update the Active table in models/fire/VERSIONS.md"
echo "   - deploy + verify on the host (deploy replaces the WHOLE remote models/:"
echo "     best.xml/best.bin/labelmap.txt must exist locally before deploying):"
echo "       ./scripts/deploy_firewatch.sh"
echo "       docker compose exec firewatch python /scripts/firewatch.py --check"
echo "       docker compose exec firewatch python /scripts/firewatch.py --dry-run"
