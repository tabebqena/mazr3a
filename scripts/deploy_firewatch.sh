#!/usr/bin/env bash
# ============================================================
# Deploy the firewatch service to the remote Frigate host.
#
# Same non-interactive password SSH (SSH_ASKPASS) pattern as
# deploy_config.sh; user/pass from .roo/rules/sshuser.md:
#   user: ai  /  host: ssh.mazr3a.garden  /  pass: 123456
#
# Uploads the files the firewatch service needs and builds/starts it:
#   docker-compose.yml        (adds the firewatch service)
#   firewatch/                (Dockerfile + requirements.txt)
#   scripts/firewatch.py, scripts/monitor_lib.py
#   config/firewatch.conf     (tunables - never touches telegram.conf)
#   models/fire/              ACTIVE model only - see "MODEL HANDLING"
#
# The git-ignored config/telegram.conf is NOT overwritten - the host copy
# (already used by the machine-monitor/machine-status crons) is reused.
#
# MODEL HANDLING (keeps a running host safe)
# ------------------------------------------
# - Only the ACTIVE model (best.xml + best.bin + labelmap.txt [+ best.pt]) is
#   ever pushed into ${REMOTE_DIR}/models/fire/ - never the whole local
#   models/ tree (so the versioned archive models/fire/versions/ stays local
#   and a host model is never wiped by a wholesale directory replace).
# - Each file is pushed only when its md5 differs from the host copy, and is
#   swapped atomically (.new -> mv) so a live firewatch never sees a
#   half-written model.
# - If the local ACTIVE OpenVINO IR is INCOMPLETE but the host already runs a
#   model, the host model is LEFT UNTOUCHED (firewatch keeps running) and the
#   deploy only warns - code/config still deploy.
# - If no model exists on either side, firewatch starts but exits until an IR
#   is provided (see models/fire/README.md).
# - frigate/mqtt are NEVER restarted by this script; only the firewatch
#   container is restarted when its model/code/config changed.
#
# Verify after deploy: --check (config + model load) then --dry-run (one live
# pass, prints instead of sending).
# ============================================================
set -euo pipefail

SSH_USER="ai"
SSH_HOST="ssh.mazr3a.garden"
SSH_PASS="123456"
REMOTE_DIR="/home/dr/frigate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- non-interactive password via SSH_ASKPASS ---------------------------
ASKPASS="$(mktemp)"
trap 'rm -f "$ASKPASS"' EXIT
printf '#!/usr/bin/env bash\necho "%s"\n' "$SSH_PASS" > "$ASKPASS"
chmod 700 "$ASKPASS"

export SSH_ASKPASS="$ASKPASS"
export SSH_ASKPASS_REQUIRE=force   # use askpass even without a tty
export DISPLAY=:0

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o NumberOfPasswordPrompts=1
  -o ConnectTimeout=20
)

# host_cmd: run a remote shell command (askpass handles the auth, no tty).
host_cmd() {
  setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$@"
}

# rput <local> <remote-abs>: scp to a .new name, then mv over the target
# (rename only needs write on the parent dir - same trick as deploy_config.sh).
rput() {
  local local_file="$1" remote_file="$2"
  setsid /usr/bin/scp "${SSH_OPTS[@]}" "$local_file" \
    "${SSH_USER}@${SSH_HOST}:${remote_file}.new"
  host_cmd "mv -f ${remote_file}.new ${remote_file}"
}

# rput_dir <local-dir> <remote-dir>: recursive upload with atomic swap.
rput_dir() {
  local local_dir="$1" remote_dir="$2"
  setsid /usr/bin/scp -r "${SSH_OPTS[@]}" "$local_dir" \
    "${SSH_USER}@${SSH_HOST}:${remote_dir}.new"
  host_cmd "rm -rf ${remote_dir} && mv ${remote_dir}.new ${remote_dir}"
}

MODEL_DIR="${ROOT_DIR}/models/fire"
MODEL_REMOTE="${REMOTE_DIR}/models/fire"

# local_md5 / remote_md5: md5 of a file, empty string when missing. Never abort.
local_md5()  { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || true; }
remote_md5() { host_cmd "md5sum '$1' 2>/dev/null | cut -d' ' -f1" || true; }

RESTART_FIREWATCH=0
MODEL_CHANGED=0

echo "=============================================================="
echo "0) sanity checks"
if [ ! -f "${ROOT_DIR}/docker-compose.yml" ]; then
  echo "ERROR: docker-compose.yml missing in ${ROOT_DIR}" >&2; exit 1
fi
if [ ! -f "${ROOT_DIR}/firewatch/Dockerfile" ]; then
  echo "ERROR: firewatch/Dockerfile missing - run from the repo root" >&2; exit 1
fi
echo "local root: ${ROOT_DIR}"
echo "remote dir: ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}"

echo "=============================================================="
echo "1) docker-compose.yml"
rput "${ROOT_DIR}/docker-compose.yml" "${REMOTE_DIR}/docker-compose.yml"

echo "=============================================================="
echo "2) firewatch/ build context (Dockerfile + requirements.txt)"
rput_dir "${ROOT_DIR}/firewatch" "${REMOTE_DIR}/firewatch"
RESTART_FIREWATCH=1

echo "=============================================================="
echo "3) scripts/firewatch.py + scripts/monitor_lib.py"
rput "${ROOT_DIR}/scripts/firewatch.py" "${REMOTE_DIR}/scripts/firewatch.py"
rput "${ROOT_DIR}/scripts/monitor_lib.py" "${REMOTE_DIR}/scripts/monitor_lib.py"
RESTART_FIREWATCH=1

echo "=============================================================="
echo "4) config/firewatch.conf (telegram.conf is never touched)"
rput "${ROOT_DIR}/config/firewatch.conf" "${REMOTE_DIR}/config/firewatch.conf"
RESTART_FIREWATCH=1

echo "=============================================================="
echo "5) ACTIVE fire model -> ${MODEL_REMOTE} (conditional)"

# firewatch needs best.xml + best.bin + labelmap.txt; best.pt is optional
# (source checkpoint kept for later re-conversion).
LOCAL_IR_OK=1
for f in best.xml best.bin labelmap.txt; do
  [ -f "${MODEL_DIR}/${f}" ] || LOCAL_IR_OK=0
done

if [ "$LOCAL_IR_OK" -eq 1 ]; then
  host_cmd "mkdir -p ${MODEL_REMOTE}"
  for f in best.xml best.bin labelmap.txt best.pt; do
    lf="${MODEL_DIR}/${f}"
    [ -f "$lf" ] || continue     # best.pt is optional on the host
    lm="$(local_md5 "$lf")"
    rmf="$(remote_md5 "${MODEL_REMOTE}/${f}")"
    if [ "$lm" != "$rmf" ]; then
      echo "   ${f}: differs from host -> push (atomic .new swap)"
      setsid /usr/bin/scp "${SSH_OPTS[@]}" "$lf" \
        "${SSH_USER}@${SSH_HOST}:${MODEL_REMOTE}/${f}.new"
      host_cmd "mv -f ${MODEL_REMOTE}/${f}.new ${MODEL_REMOTE}/${f}"
      MODEL_CHANGED=1
    else
      echo "   ${f}: unchanged -> skip"
    fi
  done
  if [ "$MODEL_CHANGED" -eq 1 ]; then
    echo "   host model updated -> firewatch restart will load it"
    RESTART_FIREWATCH=1
  else
    echo "   host already runs this ACTIVE model (no model restart needed)"
  fi
else
  echo "   WARNING: local ACTIVE OpenVINO IR is INCOMPLETE"
  echo "   (need best.xml + best.bin + labelmap.txt under models/fire/)."
  echo "   Generate/place it first, e.g.:"
  echo "     ./scripts/prep_fire_model.sh models/fire/best.pt 640 \"fire,other,smoke\""
  if remote_md5 "${MODEL_REMOTE}/best.xml" | grep -q .; then
    echo "   The HOST still runs its existing model - left untouched so a live"
    echo "   firewatch keeps working. Deploying code/config only."
  else
    echo "   The HOST has no fire model either - firewatch will start but cannot"
    echo "   load a model until an IR is provided (see models/fire/README.md)."
  fi
fi

echo "=============================================================="
echo "6) build + start firewatch (first build pulls pip deps over network)"
# `up -d --build` only (re)creates firewatch - unchanged frigate/mqtt stay up.
# A restart is forced afterwards when the model/code/config changed so the
# container re-reads the new weights/config.
host_cmd "cd ${REMOTE_DIR} && docker compose up -d --build firewatch"
if [ "$RESTART_FIREWATCH" -eq 1 ]; then
  echo "   restarting firewatch to load updated model/code/config"
  host_cmd "cd ${REMOTE_DIR} && docker compose restart firewatch" || true
fi
host_cmd "cd ${REMOTE_DIR} && sleep 8 && docker compose ps firewatch"

echo "=============================================================="
echo "7) verify: --check (config + model load)"
host_cmd "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /scripts/firewatch.py --check" || true

echo "=============================================================="
echo "8) verify: --dry-run (one live pass, prints instead of sending)"
host_cmd "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /scripts/firewatch.py --dry-run" || true

echo "=============================================================="
echo "9) recent firewatch logs"
host_cmd "cd ${REMOTE_DIR} && docker compose logs --since 3m firewatch 2>&1 | tail -30" || true

echo "=============================================================="
echo "DEPLOY COMPLETE"
echo "If a model was promoted, flip its status in models/fire/VERSIONS.md and its"
echo "VERSION.json. Confirm the --dry-run output above before trusting alerts."
