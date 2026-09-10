#!/usr/bin/env bash
# run_ssh.sh - run ONE remote command on ssh.mazr3a.garden (read-only diagnosis).
#
# Non-interactive password auth via OpenSSH SSH_ASKPASS (NO sshpass - see
# .roo/rules/ssh-password.md). Mirrors run_ssh() in dev_scripts/deploy_all.sh.
#
# The AI edit user is `ai` (see .roo/rules/sshuser.md). This helper is for
# READ-ONLY host diagnosis only: it never runs git on the host and never deploys
# (deploy/git handoff is owned by deploy_all.sh as `dr`).
#
# Usage:
#   DEPLOY_SSH_USER=ai DEPLOY_SSH_PASS='...' dev_scripts/run_ssh.sh '<remote cmd>'
#
# Credentials may come from DEPLOY_SSH_USER/DEPLOY_SSH_PASS (deploy_all.sh names)
# or SSH_USER/SSH_PASSWORD. Retries flaky tunnel connections up to 5x.
set -euo pipefail

SSH_HOST="${DEPLOY_SSH_HOST:-ssh.mazr3a.garden}"
SSH_USER="${DEPLOY_SSH_USER:-${SSH_USER:-}}"
SSH_PASS="${DEPLOY_SSH_PASS:-${SSH_PASSWORD:-}}"

if [ -z "$SSH_USER" ] || [ -z "$SSH_PASS" ]; then
  echo "ERROR: set DEPLOY_SSH_USER and DEPLOY_SSH_PASS (or SSH_USER/SSH_PASSWORD)" >&2
  exit 1
fi
if [ "$#" -lt 1 ]; then
  echo "usage: $0 '<remote command>'" >&2
  exit 1
fi

# --- SSH_ASKPASS setup (verbatim password file, no shell re-interpretation) --
PASSFILE="$(mktemp)"
printf '%s\n' "$SSH_PASS" > "$PASSFILE"
chmod 600 "$PASSFILE"
ASKPASS="$(mktemp)"
printf '#!/usr/bin/env bash\ncat "%s"\n' "$PASSFILE" > "$ASKPASS"
chmod 700 "$ASKPASS"
trap 'rm -f "$ASKPASS" "$PASSFILE"' EXIT

export SSH_ASKPASS="$ASKPASS"
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o NumberOfPasswordPrompts=1
  -o ConnectTimeout=25
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=6
)

n=1
until setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$1"; do
  echo "   (ssh retry ${n}/5)" >&2
  if [ "$n" -ge 5 ]; then
    echo "ERROR: ssh failed after 5 attempts: $1" >&2
    exit 1
  fi
  n=$((n + 1)); sleep 10
done
