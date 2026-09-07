#!/usr/bin/env bash
# ============================================================
# deploy_all.sh - GIT-BASED full deploy orchestrator.
#
# Replaces the old dev_scripts/deploy_config.sh +
# dev_scripts/deploy_firewatch.sh shims with ONE script that ALWAYS
# runs every deploy step on the host (configs + firewatch + model).
# Deploy = SSH to the Frigate host and `git pull --ff-only`, i.e. SYNC THE HOST
# to the latest REMOTE (origin/master). This script does NOT push - you run
# `git push origin master` yourself first. If there are local changes
# (uncommitted files or unpushed commits) it warns, lists them, and asks you to
# confirm (complete the deploy / abort) before touching the host.
#
#   git push origin master && deploy_all.sh   # deploy the remote state
#   deploy_all.sh                             # same - prompts if local changes
#
# There are NO `config` / `firewatch` / `bootstrap` options anymore
# (2026-09-05): config and firewatch are always both deployed, and the
# host bootstrap is done by the user, not this script.
#
# HOST PRE-BOOTSTRAP (required once, by hand - NOT this script)
# ------------------------------------------------------------
# /home/dr/frigate is the live deploy dir and must ALREADY be a git clone
# of origin/master before deploy_all.sh is run (the user bootstraps it,
# so no bootstrap option exists here). Git-ignored host state is NEVER
# touched by deploy: .env, config/telegram.conf, media/, mosquitto
# data/log, models/fire/versions/.
#
# REMOTE GIT
# ----------
#   origin = https://github.com/tabebqena/mazr3a   (branch: master)
# The local repo is pushed to origin by YOU (not this script - GitHub auth is
# interactive); the host clone pulls from it. If local has uncommitted changes
# or unpushed commits the script warns + asks to continue (host syncs to remote,
# so such local state would NOT be deployed) or abort.
#
# WHY GIT (decision 2026-09-05): hand-rolled file sync is abandoned; git
# already solves move/delete/rename natively and carries the ACTIVE fire
# model (models/fire/best.xml/bin/pt/labelmap.txt are now TRACKED - see
# .gitignore; the versions/ archive stays ignored).
#
# TRANSPORT
# ---------
# SSH to the deploy host (default ssh.mazr3a.garden, override with
# DEPLOY_SSH_HOST) through a flaky cloudflared tunnel - every ssh retries with
# keep-alives. Credentials come from the environment (DEPLOY_SSH_USER /
# DEPLOY_SSH_PASS) or are asked interactively; nothing is hard-coded anymore.
#
# OWNERSHIP: the SSH user MUST own the host clone (/home/dr/frigate). git >= 2.35.2
# refuses to run in a repo owned by another user ("detected dubious ownership")
# and group write access is NOT enough. This deploy deliberately does NOT add a
# git safe.directory for the clone - instead it runs git as the clone owner. On
# this host the clone is owned by dr, so deploy as dr (not ai).
#
# RESTARTS are driven by the changed set (old host HEAD..new HEAD) so an
# unchanged deploy restarts nothing. Verification runs at the end.
# ============================================================
set -euo pipefail

SSH_HOST="${DEPLOY_SSH_HOST:-ssh.mazr3a.garden}"
REMOTE_DIR="/home/dr/frigate"
GIT_REMOTE="https://github.com/tabebqena/mazr3a"
GIT_BRANCH="master"

# --- SSH credentials: environment first, else interactive prompt ---------
# Read from the environment (DEPLOY_SSH_USER / DEPLOY_SSH_PASS; the plain
# SSH_USER / SSH_PASSWORD names are honoured too) or ask interactively.
# The SSH user must OWN the host clone (see TRANSPORT header above): git >= 2.35.2
# otherwise aborts with "dubious ownership". We deliberately do NOT mark the
# clone as a git safe.directory - run as the clone owner instead (here: dr).
SSH_USER="${DEPLOY_SSH_USER:-${SSH_USER:-}}"
SSH_PASS="${DEPLOY_SSH_PASS:-${SSH_PASSWORD:-}}"
resolve_ssh_credentials() {
  if [ -z "$SSH_USER" ] || [ -z "$SSH_PASS" ]; then
    if [ ! -t 0 ]; then
      echo "ERROR: SSH credentials not set. Export DEPLOY_SSH_USER and DEPLOY_SSH_PASS" >&2
      echo "  (DEPLOY_SSH_USER must own ${REMOTE_DIR} on the host, e.g. dr) or run" >&2
      echo "  in an interactive terminal to be asked. Aborting." >&2
      exit 1
    fi
  fi
  if [ -z "$SSH_USER" ]; then
    printf 'SSH user for %s: ' "$SSH_HOST" >&2
    IFS= read -r SSH_USER || { echo >&2; exit 1; }
    export SSH_USER
  fi
  if [ -z "$SSH_PASS" ]; then
    printf 'SSH password for %s@%s: ' "$SSH_USER" "$SSH_HOST" >&2
    IFS= read -r -s SSH_PASS || { echo >&2; exit 1; }
    echo >&2
    export SSH_PASS
  fi
}
resolve_ssh_credentials

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- non-interactive password via SSH_ASKPASS ---------------------------
# The askpass helper cats a password file (chmod 600) so whatever was typed or
# exported is passed to ssh verbatim - no shell re-interpretation of the value.
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

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o NumberOfPasswordPrompts=1
  -o ConnectTimeout=25
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=6
)

run_ssh() { # "<remote-command>": retry up to 5x
  local cmd="$1" n=1
  until setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$cmd"; do
    echo "   (ssh retry ${n}/5)" >&2
    if [ "$n" -ge 5 ]; then echo "ERROR: ssh failed after 5 attempts: $cmd" >&2; return 1; fi
    n=$((n + 1)); sleep 10
  done
  return 0
}

# remote git helper (runs a git command inside the deploy dir on the host)
remote_git() { # <git-args...>
  run_ssh "cd ${REMOTE_DIR} && git $*"
}

confirm_proceed() { # asks once; DEPLOY_ASSUME_YES=1 skips (non-interactive use)
  if [ "${DEPLOY_ASSUME_YES:-0}" = "1" ]; then return 0; fi
  if [ ! -t 0 ]; then
    echo "ERROR: cannot prompt (stdin is not a TTY). Run in an interactive" >&2
    echo "  terminal, or set DEPLOY_ASSUME_YES=1 to auto-confirm and re-run." >&2
    exit 1
  fi
  local ans
  read -r -p "Complete the deploy (sync the HOST to origin/${GIT_BRANCH})? [y/N] " ans
  case "$ans" in
    y|Y|yes|Yes|YES) return 0 ;;
    *) echo "aborted - host not changed."; exit 1 ;;
  esac
}

check_local_and_confirm() {
  # Never pushes: GitHub auth is interactive (token/ksshaskpass). This deploy
  # SYNCS THE HOST to the latest REMOTE (origin/master). If there are local
  # changes that would NOT reach the host, warn and ask the user to complete
  # the deploy or abort.
  echo "=============================================================="
  echo "preflight: local repo vs origin/${GIT_BRANCH}"
  echo "  (this deploy syncs the HOST to origin/${GIT_BRANCH}, not to local HEAD)"
  if ! git -C "$ROOT_DIR" remote | grep -qx origin; then
    git -C "$ROOT_DIR" remote add origin "$GIT_REMOTE"
    echo "   added local origin ${GIT_REMOTE}"
  else
    git -C "$ROOT_DIR" remote set-url origin "$GIT_REMOTE"
  fi

  # 1) uncommitted working-tree changes
  if [ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]; then
    echo "WARNING: local working tree has uncommitted changes - these will NOT be deployed." >&2
    git -C "$ROOT_DIR" status --short | sed 's/^/    /' >&2
    confirm_proceed || exit 1
  fi

  # 2) local commits not yet pushed to origin (they would be left out of the host sync)
  local ahead
  ahead="$(git -C "$ROOT_DIR" rev-list --count "origin/${GIT_BRANCH}..HEAD" 2>/dev/null || echo ERR)"
  if [ "$ahead" = "ERR" ]; then
    echo "ERROR: cannot compare HEAD with origin/${GIT_BRANCH} (origin ref unknown?)." >&2
    echo "  Run: git fetch origin && git push origin ${GIT_BRANCH}" >&2
    exit 1
  fi
  if [ "$ahead" -ne 0 ]; then
    echo "WARNING: ${ahead} local commit(s) are NOT on origin/${GIT_BRANCH} yet and" >&2
    echo "  will NOT be on the host after this deploy (host syncs to origin/${GIT_BRANCH})." >&2
    git -C "$ROOT_DIR" log "origin/${GIT_BRANCH}..HEAD" --oneline | sed 's/^/    /' >&2
    echo "  origin/${GIT_BRANCH} is at $(git -C "$ROOT_DIR" rev-parse --short "origin/${GIT_BRANCH}")." >&2
    echo "  To include them: git push origin ${GIT_BRANCH} first, then re-run." >&2
    confirm_proceed || exit 1
  else
    echo "   ok: local HEAD $(git -C "$ROOT_DIR" rev-parse --short HEAD) == origin/${GIT_BRANCH}"
  fi
}

deploy() {
  echo "=============================================================="
  echo "0) preflight: local changes check (host syncs to origin/${GIT_BRANCH})"
  check_local_and_confirm

  echo "=============================================================="
  echo "0b) reachability + host repo state"
  run_ssh "mkdir -p ${REMOTE_DIR}" || exit 1

  # NOTE: make the probe always exit 0 so run_ssh does not retry a "no .git"
  host_clone="$(run_ssh "cd ${REMOTE_DIR} && if [ -d .git ]; then echo yes; else echo no; fi")"
  if [ "$host_clone" != "yes" ]; then
    echo "ERROR: ${REMOTE_DIR} on the host is NOT a git clone." >&2
    echo "  The host must be bootstrapped first (by you): git init + remote add" >&2
    echo "  origin ${GIT_REMOTE} + fetch + hard reset to origin/${GIT_BRANCH}." >&2
    exit 1
  fi

  echo "=============================================================="
  echo "1) capture host HEAD before pull"
  OLD_HEAD="$(remote_git rev-parse HEAD || true)"
  echo "   old host HEAD: ${OLD_HEAD:-none}"

  echo "=============================================================="
  echo "2) pull latest (fast-forward only) on the host"
  remote_git pull --ff-only origin "$GIT_BRANCH" || {
    echo "ERROR: git pull failed (non-fast-forward?). If the host has local"
    echo "commits/edits on master, resolve or reset them first." >&2
    exit 1
  }
  NEW_HEAD="$(remote_git rev-parse HEAD || true)"
  echo "   new host HEAD: ${NEW_HEAD:-none}"

  # determine the changed set
  declare -a CHANGED=()
  if [ -n "$OLD_HEAD" ] && [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
    while IFS= read -r f; do [ -n "$f" ] && CHANGED+=("$f"); done \
      < <(remote_git diff --name-only "$OLD_HEAD".."$NEW_HEAD")
  fi
  echo "   changed files since last deploy: ${#CHANGED[@]}"

  COMPOSE_CHANGED=0;    FRIGATE_CFG_CHANGED=0; MQTT_CFG_CHANGED=0
  FW_CODE_CHANGED=0;    FW_BUILD_CHANGED=0;    MODEL_CHANGED=0
  for f in "${CHANGED[@]}"; do
    case "$f" in
      docker-compose.yml)                            COMPOSE_CHANGED=1 ;;
      config/config.yaml)                            FRIGATE_CFG_CHANGED=1 ;;
      models/coco/*)                                 FRIGATE_CFG_CHANGED=1 ;;
      mosquitto/config/mosquitto.conf)               MQTT_CFG_CHANGED=1 ;;
      firewatch/firewatch.py|scripts/collect_sensors.py|scripts/telegram_notify.py|config/firewatch.conf) FW_CODE_CHANGED=1 ;;
      firewatch/Dockerfile|firewatch/requirements.txt) FW_BUILD_CHANGED=1 ;;
      models/fire/*)                                 MODEL_CHANGED=1 ;;
    esac
  done

  echo "=============================================================="
  echo "3) restart services as needed"
  # --- config concerns (always deployed) ---
  if [ "$COMPOSE_CHANGED" -eq 1 ]; then
    echo "   compose changed -> up -d"
    run_ssh "cd ${REMOTE_DIR} && docker compose up -d" || true
  fi
  if [ "$FRIGATE_CFG_CHANGED" -eq 1 ] || [ "$COMPOSE_CHANGED" -eq 1 ]; then
    echo "   frigate config changed -> up -d + restart frigate"
    run_ssh "cd ${REMOTE_DIR} && docker compose up -d && sleep 5 && docker compose restart frigate && sleep 10" || true
  else
    run_ssh "cd ${REMOTE_DIR} && docker compose up -d" || true   # idempotent: starts if down
  fi
  if [ "$MQTT_CFG_CHANGED" -eq 1 ]; then
    echo "   mosquitto config changed -> restart mqtt"
    run_ssh "cd ${REMOTE_DIR} && docker compose restart mqtt" || true
  fi
  # --- firewatch concerns (always deployed) ---
  if [ "$FW_BUILD_CHANGED" -eq 1 ]; then
    echo "   firewatch image changed -> up -d --build firewatch (background)"
    run_ssh "cd ${REMOTE_DIR} && rm -f /tmp/fw_up.log && (nohup docker compose up -d --build firewatch >/tmp/fw_up.log 2>&1 &) && echo 'build started'" || exit 1
    for i in $(seq 1 90); do
      sleep 10
      state="$(run_ssh "cd ${REMOTE_DIR} && docker compose ps --format '{{.Service}}|{{.Status}}' 2>/dev/null | grep '^firewatch|' || true" || true)"
      case "$state" in
        *"Up"*) echo "   firewatch Up after ~$((i * 10))s"; break ;;
      esac
      if ! run_ssh "pgrep -f 'docker compose up -d --build firewatch' >/dev/null" >/dev/null 2>&1; then
        echo "   build ended without 'Up' - log tail:"
        run_ssh "tail -40 /tmp/fw_up.log" || true
        echo "ERROR: firewatch did not come up" >&2
        exit 1
      fi
    done
  fi
  if [ "$FW_CODE_CHANGED" -eq 1 ] || [ "$MODEL_CHANGED" -eq 1 ] || [ "$FW_BUILD_CHANGED" -eq 1 ]; then
    echo "   firewatch code/config/model changed -> restart firewatch"
    run_ssh "cd ${REMOTE_DIR} && docker compose restart firewatch" || true
  fi
  run_ssh "cd ${REMOTE_DIR} && docker compose ps firewatch" || true

  echo "=============================================================="
  echo "4) confirm host status"
  echo "   host HEAD:      $(remote_git rev-parse --short HEAD || true)"
  echo "   host clean:     $(remote_git status --porcelain | head -5 || true)"
  echo "   tracked model:  $(remote_git ls-files models/fire | tr '\n' ' ' || true)"
  if [ "${#CHANGED[@]}" -eq 0 ]; then
    echo "   (no files changed since last deploy - services untouched)"
  else
    echo "   deployed: ${CHANGED[*]}"
  fi

  echo "=============================================================="
  echo "5) verify the new configs loaded"
  run_ssh "cd ${REMOTE_DIR} && docker compose ps" || true
  run_ssh "cd ${REMOTE_DIR} && python3 scripts/verify_remote.py" || true
  echo "   -- frigate config/error log scan --"
  run_ssh "cd ${REMOTE_DIR} && docker compose logs --since 3m frigate 2>&1 | grep -iE 'invalid|error|safe mode|config' | tail -25 || true" || true

  echo "6) confirm the new model"
  run_ssh "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /firewatch/firewatch.py --check" || true
  echo "7) confirm the new scripts + one live pass"
  run_ssh "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /firewatch/firewatch.py --dry-run" || true
  echo "   -- recent firewatch logs --"
  run_ssh "cd ${REMOTE_DIR} && docker compose logs --since 5m firewatch 2>&1 | tail -20" || true

  echo "=============================================================="
  echo "DEPLOY COMPLETE (full deploy: configs + firewatch + model)"
  echo "Next local edits: commit on this machine, then re-run $0."
  echo "If a model changed, also flip its status in models/fire/VERSIONS.md + VERSION.json."
}

deploy
