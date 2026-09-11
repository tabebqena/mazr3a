#!/usr/bin/env bash
# run_rsync.sh - resume-friendly rsync to/from ssh.mazr3a.garden over the
# Cloudflare SSH tunnel, using OpenSSH SSH_ASKPASS (NO sshpass - see
# .roo/rules/ssh-password.md). Sibling of run_ssh.sh for BULK transfers.
#
# WHY: the tunnel is low-bandwidth and bursty, and run_ssh.sh restarts a
# streamed `tar` from zero after a drop. rsync resumes where it left off
# (--partial --append-verify), so a flaky link cannot lose a whole transfer.
#
# Usage:
#   DEPLOY_SSH_USER=ai DEPLOY_SSH_PASS='...' dev_scripts/run_rsync.sh <src> <dst> [rsync args...]
#
# Examples:
#   # pull the staged clip export to the local mirror (resumable)
#   DEPLOY_SSH_USER=ai DEPLOY_SSH_PASS='...' \
#     dev_scripts/run_rsync.sh ai@ssh.mazr3a.garden:/home/dr/frigate/cam-clips/ ./camera-clips/
#
# Defaults favour a large, flaky link onto an NTFS mount:
#   --partial --append-verify   resume partial files instead of restarting
#   -rlt --no-perms/owner/group  NTFS target cannot keep POSIX metadata
set -euo pipefail

SSH_HOST="${DEPLOY_SSH_HOST:-ssh.mazr3a.garden}"
SSH_USER="${DEPLOY_SSH_USER:-${SSH_USER:-}}"
SSH_PASS="${DEPLOY_SSH_PASS:-${SSH_PASSWORD:-}}"

if [ -z "$SSH_USER" ] || [ -z "$SSH_PASS" ]; then
  echo "ERROR: set DEPLOY_SSH_USER and DEPLOY_SSH_PASS (or SSH_USER/SSH_PASSWORD)" >&2
  exit 1
fi
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <src> <dst> [rsync args...]" >&2
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
export SSH_ASKPASS_REQUIRE=force   # use askpass even without a tty
export DISPLAY=:0

SSH_CMD="/usr/bin/ssh -o StrictHostKeyChecking=accept-new \
  -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=6"

exec rsync -e "$SSH_CMD" --partial --append-verify \
  -rlt --no-perms --no-owner --no-group --info=progress2 --human-readable "$@"
