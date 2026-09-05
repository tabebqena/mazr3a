#!/usr/bin/env bash
# ============================================================
# Deploy config/config.yaml to the remote Frigate host using
# password-based SSH WITHOUT prompting (no sshpass required).
#
# Uses SSH_ASKPASS (OpenSSH 8.4+) so scp/ssh get the password
# non-interactively. User/password come from .roo/rules/sshuser.md:
#   user: ai  /  host: ssh.mazr3a.garden  /  pass: 123456
# ============================================================
set -euo pipefail

SSH_USER="ai"
SSH_HOST="ssh.mazr3a.garden"
SSH_PASS="123456"
# The real Frigate project lives under the dr user's home on the host
# (verified: /home/dr/frigate/docker-compose.yml is the active compose file).
REMOTE_DIR="/home/dr/frigate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CONFIG="${SCRIPT_DIR}/../config/config.yaml"
# verify_remote.py is a HOST script (stays in scripts/); dev_scripts/ holds only
# local tooling + this deploy orchestrator, so point one level up.
LOCAL_VERIFY="${SCRIPT_DIR}/../scripts/verify_remote.py"

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

echo "=============================================================="
echo "1) scp config/config.yaml -> ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/config/config.yaml"
# The existing config.yaml is owned by dr (644); ai cannot overwrite it in
# place, but the config dir is world-writable so we scp to a temp name and
# mv -f over it (rename only needs write on the directory).
setsid /usr/bin/scp "${SSH_OPTS[@]}" \
  "$LOCAL_CONFIG" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/config/config.yaml.new"
setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "mv -f ${REMOTE_DIR}/config/config.yaml.new ${REMOTE_DIR}/config/config.yaml && ls -la ${REMOTE_DIR}/config/config.yaml"

echo "=============================================================="
echo "2) scp verify_remote.py -> ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/scripts/verify_remote.py"
setsid /usr/bin/scp "${SSH_OPTS[@]}" \
  "$LOCAL_VERIFY" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/scripts/verify_remote.py"

echo "=============================================================="
echo "3) start/recreate frigate (reloads config.yaml)"
# `up -d` (not `restart`) is intentional: it is idempotent and also
# starts the stack if it is currently down (e.g. after a host reboot),
# which `restart` silently fails to do - deploy then hangs at stage 4
# with "Connection refused" because the API never comes up.
setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd ${REMOTE_DIR} && docker compose up -d && sleep 10 && docker compose ps"

echo "=============================================================="
echo "4) verify effective object.track per camera via /api/config"
setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd ${REMOTE_DIR} && python3 scripts/verify_remote.py"

echo "=============================================================="
echo "5) scan recent frigate logs for config errors"
setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd ${REMOTE_DIR} && docker compose logs --since 3m frigate 2>&1 | grep -iE 'invalid|error|safe mode|config' | tail -25 || true"

echo "=============================================================="
echo "DEPLOY COMPLETE"
