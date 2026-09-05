#!/usr/bin/env bash
# ============================================================
# Deploy the firewatch service to the remote Frigate host.
#
# Same non-interactive password SSH (SSH_ASKPASS) pattern as
# deploy_config.sh; user/pass from .roo/rules/sshuser.md:
#   user: ai  /  host: ssh.mazr3a.garden  /  pass: 123456
#
# Uploads the files the firewatch service needs and builds/starts it:
#   docker-compose.yml  (adds the firewatch service)
#   firewatch/          (Dockerfile + requirements.txt)
#   scripts/firewatch.py, scripts/monitor_lib.py
#   config/firewatch.conf  (tunables - never touches telegram.conf)
#   models/fire/        (OpenVINO IR model: best.xml + best.bin + labelmap)
#
# The git-ignored config/telegram.conf is NOT overwritten - the host copy
# (already used by the machine-monitor/machine-status crons) is reused.
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

# rput <local> <remote-abs>: scp to a .new name, then mv over the target
# (rename only needs write on the parent dir - same trick as deploy_config.sh).
rput() {
  local local_file="$1" remote_file="$2"
  setsid /usr/bin/scp "${SSH_OPTS[@]}" "$local_file" \
    "${SSH_USER}@${SSH_HOST}:${remote_file}.new"
  setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
    "mv -f ${remote_file}.new ${remote_file}"
}

# rput_dir <local-dir> <remote-dir>: recursive upload with atomic swap.
rput_dir() {
  local local_dir="$1" remote_dir="$2"
  setsid /usr/bin/scp -r "${SSH_OPTS[@]}" "$local_dir" \
    "${SSH_USER}@${SSH_HOST}:${remote_dir}.new"
  setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
    "rm -rf ${remote_dir} && mv ${remote_dir}.new ${remote_dir}"
}

MODEL_XML="${ROOT_DIR}/models/fire/best.xml"

echo "=============================================================="
echo "0) sanity checks"
if [ ! -f "${ROOT_DIR}/docker-compose.yml" ]; then
  echo "ERROR: docker-compose.yml missing in ${ROOT_DIR}" >&2; exit 1
fi
if [ ! -f "${ROOT_DIR}/firewatch/Dockerfile" ]; then
  echo "ERROR: firewatch/Dockerfile missing - run from the repo root" >&2; exit 1
fi
if [ ! -f "${MODEL_XML}" ]; then
  echo "WARNING: ${MODEL_XML} not found - firewatch will build and start but"
  echo "         exit until the model is acquired (see models/fire/README.md)."
fi
echo "local root: ${ROOT_DIR}"
echo "remote dir: ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}"

echo "=============================================================="
echo "1) docker-compose.yml"
rput "${ROOT_DIR}/docker-compose.yml" "${REMOTE_DIR}/docker-compose.yml"

echo "=============================================================="
echo "2) firewatch/ build context (Dockerfile + requirements.txt)"
rput_dir "${ROOT_DIR}/firewatch" "${REMOTE_DIR}/firewatch"

echo "=============================================================="
echo "3) scripts/firewatch.py + scripts/monitor_lib.py"
rput "${ROOT_DIR}/scripts/firewatch.py" "${REMOTE_DIR}/scripts/firewatch.py"
rput "${ROOT_DIR}/scripts/monitor_lib.py" "${REMOTE_DIR}/scripts/monitor_lib.py"

echo "=============================================================="
echo "4) config/firewatch.conf (telegram.conf is never touched)"
rput "${ROOT_DIR}/config/firewatch.conf" "${REMOTE_DIR}/config/firewatch.conf"

echo "=============================================================="
echo "5) models/ (fire model artifacts)"
rput_dir "${ROOT_DIR}/models" "${REMOTE_DIR}/models"

echo "=============================================================="
echo "6) build + start firewatch (first build pulls pip deps over network)"
setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd ${REMOTE_DIR} && docker compose up -d --build firewatch && sleep 8 && docker compose ps"

echo "=============================================================="
echo "7) verify: container --check (config + model load)"
setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /scripts/firewatch.py --check || true"

echo "=============================================================="
echo "8) recent firewatch logs"
setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd ${REMOTE_DIR} && docker compose logs --since 3m firewatch 2>&1 | tail -30 || true"

echo "=============================================================="
echo "DEPLOY COMPLETE"
echo "Next: confirm config/telegram.conf on the host holds a real bot token"
echo "      and chat id, then run a one-off pass / live test:"
echo "      docker compose exec firewatch python /scripts/firewatch.py --dry-run"
