#!/usr/bin/env bash
# pull_camera_clips.sh - download ONLY the camera clips that are not yet local.
#
# CHECKS FIRST, downloads second:
#   1. lists the remote export (relative path + size) over SSH,
#   2. compares every file with the local mirror,
#   3. prints exactly what is missing (count + bytes, per camera),
#   4. transfers JUST those files with rsync.
#
# Because the "what is missing" set is recomputed from scratch on every run, you
# can STOP it at any time and RE-RUN it: files already present with the right size
# are skipped, and a partial/interrupted file (wrong size) is re-fetched. It does
# NOT stream the whole tree and does NOT depend on rsync's saved state.
#
# Prereq: rsync + ssh on both ends; SSH password via DEPLOY_SSH_* using OpenSSH
# SSH_ASKPASS (NO sshpass - see .roo/rules/ssh-password.md).
#
# Usage:
#   DEPLOY_SSH_USER=ai DEPLOY_SSH_PASS='...' dev_scripts/pull_camera_clips.sh [--check-only]
#
# Env (defaults in []):
#   REMOTE_DIR  remote export dir   [/home/dr/frigate/cam-clips]
#   LOCAL_DIR   local mirror        [./camera-clips]
#   SSH_HOST                        [ssh.mazr3a.garden]
#   SSH_USER / SSH_PASS   (or DEPLOY_SSH_USER / DEPLOY_SSH_PASS)
set -uo pipefail

REMOTE_DIR="${REMOTE_DIR:-/home/dr/frigate/cam-clips}"
LOCAL_DIR="${LOCAL_DIR:-./camera-clips}"
SSH_HOST="${DEPLOY_SSH_HOST:-${SSH_HOST:-ssh.mazr3a.garden}}"
SSH_USER="${DEPLOY_SSH_USER:-${SSH_USER:-}}"
SSH_PASS="${DEPLOY_SSH_PASS:-${SSH_PASSWORD:-}}"
CHECK_ONLY=0
[ "${1:-}" = "--check-only" ] && CHECK_ONLY=1

if [ -z "$SSH_USER" ] || [ -z "$SSH_PASS" ]; then
  echo "ERROR: set DEPLOY_SSH_USER and DEPLOY_SSH_PASS (or SSH_USER/SSH_PASSWORD)" >&2
  exit 1
fi
command -v rsync >/dev/null || { echo "ERROR: rsync not installed locally" >&2; exit 1; }

human() { awk -v n="${1:-0}" 'BEGIN{s="B KiB MiB GiB TiB";split(s,a," ");i=1;
  while(n>=1024 && i<5){n/=1024;i++} printf (i==1?"%d %s":"%.1f %s"), n, a[i]}'; }

# --- SSH_ASKPASS setup (verbatim password file, no shell re-interpretation) ---
PASSFILE="$(mktemp)"; printf '%s\n' "$SSH_PASS" > "$PASSFILE"; chmod 600 "$PASSFILE"
ASKPASS="$(mktemp)"
printf '#!/usr/bin/env bash\ncat "%s"\n' "$PASSFILE" > "$ASKPASS"; chmod 700 "$ASKPASS"
trap 'rm -f "$ASKPASS" "$PASSFILE"' EXIT
export SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force DISPLAY=:0

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25
  -o ServerAliveInterval=15 -o ServerAliveCountMax=6)
SSH_CMD="/usr/bin/ssh ${SSH_OPTS[*]}"

mkdir -p "$LOCAL_DIR"
LIST="$LOCAL_DIR/.missing.txt"

# --- 1) remote inventory: "relpath<TAB>size" for every file -------------------
echo "listing remote  $REMOTE_DIR  (over $SSH_USER@$SSH_HOST) ..."
REMOTE_LIST="$(setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "cd '$REMOTE_DIR' && find . -type f -printf '%P\t%s\n'" 2>/dev/null)"
if [ -z "$REMOTE_LIST" ]; then
  echo "ERROR: could not list $REMOTE_DIR on the host (ssh/rsync/path problem?)" >&2
  exit 1
fi

# --- 2) compute what is missing or the wrong size -----------------------------
: > "$LIST"
total=0 want=0 miss=0 need=0
declare -A cam_miss=() cam_need=()
while IFS=$'\t' read -r rel size; do
  [ -z "${rel:-}" ] && continue
  total=$((total + 1)); want=$((want + size))
  local_size=""
  [ -f "$LOCAL_DIR/$rel" ] && local_size="$(stat -c '%s' "$LOCAL_DIR/$rel" 2>/dev/null)"
  [ "$local_size" = "$size" ] && continue          # already present, same size
  printf '%s\n' "$rel" >> "$LIST"
  miss=$((miss + 1)); need=$((need + size))
  cam="${rel%%/*}"
  cam_miss[$cam]=$(( ${cam_miss[$cam]:-0} + 1 ))
  cam_need[$cam]=$(( ${cam_need[$cam]:-0} + size ))
done <<< "$REMOTE_LIST"

echo
echo "remote files : $total   ($(human "$want"))"
echo "local ok     : $((total - miss))"
echo "missing      : $miss   ($(human "$need"))"
if [ "$miss" -gt 0 ]; then
  echo "per camera:"
  for c in $(printf '%s\n' "${!cam_miss[@]}" | sort); do
    printf "  %-7s %6d files  %12s\n" "$c" "${cam_miss[$c]}" "$(human "${cam_need[$c]}")"
  done
  echo "(list: $LIST)"
fi

if [ "$CHECK_ONLY" = "1" ]; then
  echo "check-only: nothing downloaded."
  exit 0
fi
if [ "$miss" -eq 0 ]; then
  echo "nothing to download - local mirror is complete."
  exit 0
fi

# --- 3) transfer ONLY the missing files (single resumable stream) -------------
echo
echo "downloading $miss file(s) with rsync (resumable - Ctrl-C and re-run anytime)..."
/usr/bin/rsync -e "$SSH_CMD" --partial --files-from="$LIST" \
  -rlt --no-perms --no-owner --no-group --info=progress2 --human-readable \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/" "$LOCAL_DIR/"
rc=$?
echo
echo "rsync rc=$rc"
echo "Re-run this script to fetch anything still missing."
exit $rc
