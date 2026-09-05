#!/usr/bin/env bash
# ============================================================
# deploy_all.sh - UNIFIED GIT-BASED deploy orchestrator.
#
# Replaces dev_scripts/deploy_config.sh + dev_scripts/deploy_firewatch.sh
# with ONE script. Deploy = push the local repo to origin, then SSH to the
# Frigate host and `git pull --ff-only`. Git natively handles adds, edits,
# moves and deletes - no md5, no file-diff upload scripts.
#
#   deploy_all.sh                # full deploy (configs + firewatch + model)
#   deploy_all.sh config         # deploy all + restart/verify frigate config
#   deploy_all.sh firewatch      # deploy all + build/restart/verify firewatch
#   deploy_all.sh bootstrap      # one-time: turn /home/dr/frigate into a clone
#
# REMOTE GIT
# ----------
#   origin = https://github.com/tabebqena/mazr3a   (branch: master)
# The local repo commits/pushes to origin; the host clone pulls from it.
#
# WHY GIT (decision 2026-09-05): hand-rolled file sync is abandoned; git
# already solves move/delete/rename natively and carries the ACTIVE fire
# model (models/fire/best.xml/bin/pt/labelmap.txt are now TRACKED - see
# .gitignore; the versions/ archive stays ignored).
#
# HOST BOOTSTRAP (one-time, idempotent)
# -------------------------------------
# /home/dr/frigate is the live deploy dir but NOT a clone yet. `bootstrap`
# runs git init + remote add + fetch + hard reset to origin/master. Git-ignored
# host state is NEVER touched: .env, config/telegram.conf, media/, mosquitto
# data/log, models/fire/versions/.
#
# TRANSPORT
# ---------
# SSH to ai@ssh.mazr3a.garden (password via SSH_ASKPASS, .roo/rules/sshuser.md)
# through a flaky cloudflared tunnel - every ssh retries with keep-alives.
# ai is in group dr, so it can write /home/dr/frigate.
#
# RESTARTS are driven by the changed set (old host HEAD..new HEAD) so an
# unchanged deploy restarts nothing. Verification runs at the end.
# ============================================================
set -euo pipefail

SSH_USER="ai"
SSH_HOST="ssh.mazr3a.garden"
SSH_PASS="123456"
REMOTE_DIR="/home/dr/frigate"
GIT_REMOTE="https://github.com/tabebqena/mazr3a"
GIT_BRANCH="master"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CMD="${1:-deploy}"
case "$CMD" in
  deploy|config|firewatch|bootstrap|-h|--help|help) ;;
  *) echo "ERROR: unknown command '$CMD' (deploy|config|firewatch|bootstrap)" >&2; exit 2 ;;
esac
case "$CMD" in
  config)     SCOPE="config" ;;
  firewatch)  SCOPE="firewatch" ;;
  deploy)     SCOPE="all" ;;
  *)          SCOPE="" ;;
esac

# --- non-interactive password via SSH_ASKPASS ---------------------------
ASKPASS="$(mktemp)"
printf '#!/usr/bin/env bash\necho "%s"\n' "$SSH_PASS" > "$ASKPASS"
chmod 700 "$ASKPASS"
trap 'rm -f "$ASKPASS"' EXIT

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

bootstrap_host() {
  echo "=============================================================="
  echo "BOOTSTRAP: turn ${REMOTE_DIR} into a clone of ${GIT_REMOTE}"
  echo "   (git-ignored host state .env/telegram.conf/media/versions stays)"
  # ONE remote pass (flaky tunnel): init, origin, fetch, hard reset, branch.
  # `safe.directory` avoids git's "dubious ownership" because the repo dir is
  # dr-owned while the deploy runs as ai (ai is in group dr and can write).
  run_ssh "set -e
mkdir -p ${REMOTE_DIR}
git config --global --add safe.directory ${REMOTE_DIR} 2>/dev/null || true
cd ${REMOTE_DIR}
if [ ! -d .git ]; then git init -q; fi
if git remote | grep -qx origin; then git remote set-url origin ${GIT_REMOTE}; else git remote add origin ${GIT_REMOTE}; fi
git fetch origin ${GIT_BRANCH}
git reset --hard origin/${GIT_BRANCH}
git branch -M ${GIT_BRANCH}
git branch --set-upstream-to=origin/${GIT_BRANCH} ${GIT_BRANCH} 2>/dev/null || true
echo 'bootstrap: OK'" || exit 1
  echo "   host now at: $(remote_git rev-parse --short HEAD || echo '?')"
  echo "BOOTSTRAP COMPLETE"
}

push_local() {
  echo "=============================================================="
  echo "push local ${GIT_BRANCH} -> origin"
  # ensure origin is configured locally too
  if ! git -C "$ROOT_DIR" remote | grep -qx origin; then
    git -C "$ROOT_DIR" remote add origin "$GIT_REMOTE"
    echo "   added local origin ${GIT_REMOTE}"
  else
    git -C "$ROOT_DIR" remote set-url origin "$GIT_REMOTE"
  fi
  # refuse to push uncommitted work: Agents.md rule = commit after each change
  if [ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]; then
    echo "ERROR: local working tree has uncommitted changes - commit first" >&2
    echo "  (git add -A && git commit -m '...' && $0 $CMD)" >&2
    exit 1
  fi
  git -C "$ROOT_DIR" push -u origin "$GIT_BRANCH"
  echo "   pushed $(git -C "$ROOT_DIR" rev-parse --short HEAD)"
}

deploy() {
  echo "=============================================================="
  echo "0) preflight: reachability + host repo state  (scope=${SCOPE:-bootstrap})"
  run_ssh "mkdir -p ${REMOTE_DIR}" || exit 1

  # push FIRST so a fresh/empty remote has a master branch to fetch, and so a
  # host that is not a clone yet can bootstrap to exactly our pushed HEAD.
  push_local

  BOOTSTRAPPED=0
  # NOTE: make the probe always exit 0 so run_ssh does not retry a "no .git"
  host_clone="$(run_ssh "cd ${REMOTE_DIR} && if [ -d .git ]; then echo yes; else echo no; fi")"
  if [ "$host_clone" = "yes" ]; then
    echo "   host is already a git clone"
  else
    echo "   host is NOT a clone -> running bootstrap first"
    bootstrap_host
    BOOTSTRAPPED=1
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

  # determine the changed set (relative to the scope's concern)
  declare -a CHANGED=()
  if [ -n "$OLD_HEAD" ] && [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
    while IFS= read -r f; do [ -n "$f" ] && CHANGED+=("$f"); done \
      < <(remote_git diff --name-only "$OLD_HEAD".."$NEW_HEAD")
  fi
  if [ "$BOOTSTRAPPED" -eq 1 ]; then
    # first-time clone: no previous state to diff -> treat the whole stack as
    # needing a fresh start (compose up + frigate restart + firewatch build/start).
    echo "   (first-time bootstrap: no prior HEAD to diff - will start the stack fresh)"
  fi
  echo "   changed files since last deploy: ${#CHANGED[@]}"

  in_scope() { # <mask> - true when current scope covers this concern
    case ",$1," in *",$SCOPE,"*|*",all,"*) return 0 ;; *) return 1 ;; esac
  }

  COMPOSE_CHANGED=0;    FRIGATE_CFG_CHANGED=0; MQTT_CFG_CHANGED=0
  FW_CODE_CHANGED=0;    FW_BUILD_CHANGED=0;    MODEL_CHANGED=0
  if [ "$BOOTSTRAPPED" -eq 1 ]; then
    # first-time clone -> assume everything changed (scope-aware below)
    COMPOSE_CHANGED=1; FRIGATE_CFG_CHANGED=1; MQTT_CFG_CHANGED=1
    FW_CODE_CHANGED=1; FW_BUILD_CHANGED=1;    MODEL_CHANGED=1
    CHANGED+=("(bootstrap - full restart)")
  fi
  for f in "${CHANGED[@]}"; do
    case "$f" in
      docker-compose.yml)                            COMPOSE_CHANGED=1 ;;
      config/config.yaml)                            FRIGATE_CFG_CHANGED=1 ;;
      mosquitto/config/mosquitto.conf)               MQTT_CFG_CHANGED=1 ;;
      scripts/firewatch.py|scripts/collect_sensors.py|config/firewatch.conf) FW_CODE_CHANGED=1 ;;
      firewatch/Dockerfile|firewatch/requirements.txt) FW_BUILD_CHANGED=1 ;;
      models/fire/*)                                 MODEL_CHANGED=1 ;;
    esac
  done

  echo "=============================================================="
  echo "3) restart services as needed"
  # config concerns (scope: all|config)
  if in_scope "all,config"; then
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
  fi
  # firewatch concerns (scope: all|firewatch)
  if in_scope "all,firewatch"; then
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
  fi

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
  if in_scope "all,config"; then
    echo "5) verify the new configs loaded"
    run_ssh "cd ${REMOTE_DIR} && docker compose ps" || true
    run_ssh "cd ${REMOTE_DIR} && python3 scripts/verify_remote.py" || true
    echo "   -- frigate config/error log scan --"
    run_ssh "cd ${REMOTE_DIR} && docker compose logs --since 3m frigate 2>&1 | grep -iE 'invalid|error|safe mode|config' | tail -25 || true" || true
  fi

  if in_scope "all,firewatch"; then
    echo "6) confirm the new model"
    run_ssh "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /scripts/firewatch.py --check" || true
    echo "7) confirm the new scripts + one live pass"
    run_ssh "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /scripts/firewatch.py --dry-run" || true
    echo "   -- recent firewatch logs --"
    run_ssh "cd ${REMOTE_DIR} && docker compose logs --since 5m firewatch 2>&1 | tail -20" || true
  fi

  echo "=============================================================="
  echo "DEPLOY COMPLETE (command=$CMD)"
  echo "Next local edits: commit on this machine, then re-run $0 $CMD."
  echo "If a model changed, also flip its status in models/fire/VERSIONS.md + VERSION.json."
}

if [ "$CMD" = "bootstrap" ]; then
  echo "reachability check..."
  run_ssh "mkdir -p ${REMOTE_DIR}" || exit 1
  bootstrap_host
  echo "now run: $0 deploy    (or: $0 config / $0 firewatch)"
  exit 0
fi

deploy
