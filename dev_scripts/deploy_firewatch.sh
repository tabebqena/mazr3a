#!/usr/bin/env bash
# ============================================================
# Deploy the firewatch service to the remote Frigate host.
#
# Same non-interactive password SSH (SSH_ASKPASS) pattern as
# deploy_config.sh; user/pass from .roo/rules/sshuser.md:
#   user: ai  /  host: ssh.mazr3a.garden  /  pass: 123456
#
# TRANSPORT (robust against a flaky tunnel)
# ----------------------------------------
# SSH/scp to this host travels through a Cloudflare Access tunnel
# (cloudflared), where many small sequential connections stall. This deploy
# therefore does ONE upload (a single tarball of everything firewatch needs)
# + ONE docker `install.sh` pass that writes every file as root, and only then
# builds/starts the service. Every ssh/scp retries and sends keep-alives.
#
# WHY ROOT WRITES
# ---------------
# User `ai` is NOT allowed to write the top-level project dir /home/dr/frigate
# (owner dr, ACL user:ai:r-x), but IS in the docker group - so uploads are
# staged under world-writable config/.deploy_stage/ and installed as root via
# a bind-mounted `alpine` container. Files are replaced atomically (.new->mv).
#
# MODEL HANDLING (keeps a running host safe)
# ------------------------------------------
# - Only the ACTIVE model (best.xml + best.bin + labelmap.txt [+ best.pt]) is
#   bundled into ${REMOTE_DIR}/models/fire/ - never the local models/ tree
#   (the versioned archive models/fire/versions/ stays local; nothing on the
#   host is ever deleted by install.sh - files absent from the bundle are left
#   alone, so a working host model is never erased).
# - If the local ACTIVE OpenVINO IR is INCOMPLETE but the host runs one, the
#   model files are NOT bundled at all and the host model is left untouched.
# - If neither side has an IR, firewatch starts but exits until one is
#   provided (see models/fire/README.md).
# - frigate/mqtt are NEVER restarted; only the firewatch container is restarted
#   when the model/code/config changed.
#
# Verify after deploy: --check (config + model load) then --dry-run (one live
# pass, prints instead of sending).
# ============================================================
set -euo pipefail

SSH_USER="ai"
SSH_HOST="ssh.mazr3a.garden"
SSH_PASS="123456"
REMOTE_DIR="/home/dr/frigate"
STAGE_DIR="/home/dr/frigate/config/.deploy_stage"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUNDLE="$(mktemp -d)"
TARBALL="$(mktemp -u /tmp/firewatch_deploy.XXXXXX).tar.gz"
trap 'rm -rf "$BUNDLE" "$TARBALL"' EXIT

# --- non-interactive password via SSH_ASKPASS ---------------------------
ASKPASS="$(mktemp)"
printf '#!/usr/bin/env bash\necho "%s"\n' "$SSH_PASS" > "$ASKPASS"
chmod 700 "$ASKPASS"
trap 'rm -f "$ASKPASS"; rm -rf "$BUNDLE" "$TARBALL"' EXIT

export SSH_ASKPASS="$ASKPASS"
export SSH_ASKPASS_REQUIRE=force   # use askpass even without a tty
export DISPLAY=:0

# keep-alives + retries: the cloudflared tunnel drops idle/long connections.
SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o NumberOfPasswordPrompts=1
  -o ConnectTimeout=25
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=6
)

# run_ssh "<remote-command>": retry up to 5x. Returns 0 on success.
run_ssh() {
  local cmd="$1" n=1
  until setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$cmd"; do
    echo "   (ssh retry ${n}/5)" >&2
    if [ "$n" -ge 5 ]; then echo "ERROR: ssh failed after 5 attempts: $cmd" >&2; return 1; fi
    n=$((n + 1)); sleep 10
  done
  return 0
}

# run_scp <local> <remote-abs>: retry up to 5x.
run_scp() {
  local src="$1" dst="$2" n=1
  until setsid /usr/bin/scp "${SSH_OPTS[@]}" "$src" "${SSH_USER}@${SSH_HOST}:$dst"; do
    echo "   (scp retry ${n}/5)" >&2
    if [ "$n" -ge 5 ]; then echo "ERROR: scp failed after 5 attempts: $src" >&2; return 1; fi
    n=$((n + 1)); sleep 10
  done
  return 0
}

MODEL_DIR="${ROOT_DIR}/models/fire"
MODEL_REMOTE="${REMOTE_DIR}/models/fire"
local_md5() { md5sum "$1" 2>/dev/null | cut -d' ' -f1 || true; }

RESTART_FIREWATCH=0
MODEL_CHANGED=0

echo "=============================================================="
echo "0) sanity checks + remote reachability"
[ -f "${ROOT_DIR}/docker-compose.yml" ]  || { echo "ERROR: docker-compose.yml missing" >&2; exit 1; }
[ -f "${ROOT_DIR}/firewatch/Dockerfile" ] || { echo "ERROR: firewatch/Dockerfile missing" >&2; exit 1; }
echo "local root:  ${ROOT_DIR}"
echo "remote dir:  ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}"
run_ssh "mkdir -p ${STAGE_DIR}" || exit 1

echo "=============================================================="
echo "1) assemble deploy bundle (single upload)"

# --- regular files -------------------------------------------------------
cp "${ROOT_DIR}/docker-compose.yml"          "$BUNDLE/docker-compose.yml"
cp -r "${ROOT_DIR}/firewatch"                "$BUNDLE/firewatch"
mkdir -p "$BUNDLE/scripts" "$BUNDLE/config"
cp "${ROOT_DIR}/scripts/firewatch.py"        "$BUNDLE/scripts/firewatch.py"
cp "${ROOT_DIR}/scripts/collect_sensors.py"  "$BUNDLE/scripts/collect_sensors.py"
cp "${ROOT_DIR}/config/firewatch.conf"       "$BUNDLE/config/firewatch.conf"
RESTART_FIREWATCH=1

# --- active model: bundle ONLY when the local IR is complete, and only when
#     it differs from what the host already runs --------------------------
LOCAL_IR_OK=1
for f in best.xml best.bin labelmap.txt; do
  [ -f "${MODEL_DIR}/${f}" ] || LOCAL_IR_OK=0
done
if [ "$LOCAL_IR_OK" -eq 1 ]; then
  # sentinel check: does the host already run this exact best.xml?
  remote_xml="$(run_ssh "md5sum ${MODEL_REMOTE}/best.xml 2>/dev/null | cut -d' ' -f1" || true)"
  if [ "$(local_md5 "${MODEL_DIR}/best.xml")" != "$remote_xml" ]; then
    mkdir -p "$BUNDLE/models/fire"
    for f in best.xml best.bin labelmap.txt best.pt; do
      [ -f "${MODEL_DIR}/${f}" ] && cp "${MODEL_DIR}/${f}" "$BUNDLE/models/fire/${f}"
    done
    MODEL_CHANGED=1
    RESTART_FIREWATCH=1
    echo "   model bundled (host differs / missing)"
  else
    echo "   host already runs this ACTIVE model - not re-uploaded"
  fi
else
  echo "   WARNING: local ACTIVE OpenVINO IR INCOMPLETE (need best.xml+best.bin+"
  echo "   labelmap.txt under models/fire/). If the host runs a model it is left"
  echo "   untouched. Generate IR first: ./dev_scripts/prep_fire_model.sh models/fire/best.pt 640 \"fire,other,smoke\""
fi

# --- install.sh (runs as root inside the bind-mounted alpine) ------------
cat > "$BUNDLE/install.sh" <<'INSTALL'
#!/bin/sh
# Installs the firewatch deploy bundle into $1 (the project root).
# POSIX sh, busybox-safe. Writes are atomic (.new -> mv). NEVER deletes
# anything on the host except the firewatch build dir it is about to replace.
set -e
root="$1"
src="$root/config/.deploy_extract"

# 1) docker-compose.yml (atomic replace)
if [ -f "$src/docker-compose.yml" ]; then
  cat "$src/docker-compose.yml" > "$root/docker-compose.yml.new"
  mv -f "$root/docker-compose.yml.new" "$root/docker-compose.yml"
fi

# 2) firewatch/ build context (replace wholesale - it is only a build context)
if [ -d "$src/firewatch" ]; then
  rm -rf "$root/firewatch"
  mv "$src/firewatch" "$root/firewatch"
fi

# 3) scripts (firewatch.py + collect_sensors.py)
if [ -f "$src/scripts/firewatch.py" ]; then
  mkdir -p "$root/scripts"
  cat "$src/scripts/firewatch.py" > "$root/scripts/firewatch.py"
fi
if [ -f "$src/scripts/collect_sensors.py" ]; then
  mkdir -p "$root/scripts"
  cat "$src/scripts/collect_sensors.py" > "$root/scripts/collect_sensors.py"
fi

# 4) config/firewatch.conf ONLY (telegram.conf is never touched)
if [ -f "$src/config/firewatch.conf" ]; then
  mkdir -p "$root/config"
  cat "$src/config/firewatch.conf" > "$root/config/firewatch.conf"
fi

# 5) ACTIVE model files only - files NOT in the bundle are left untouched
if [ -d "$src/models" ]; then
  for f in "$src"/models/*/*; do
    [ -f "$f" ] || continue
    rel="${f#$src/}"
    mkdir -p "$root/$(dirname "$rel")"
    cat "$f" > "$root/$rel"
  done
fi
echo "install.sh: OK"
INSTALL
chmod +x "$BUNDLE/install.sh"

tar -C "$BUNDLE" -czf "$TARBALL" .

echo "=============================================================="
echo "2) upload bundle (single scp)"
run_scp "$TARBALL" "${STAGE_DIR}/deploy.tar.gz" || exit 1

echo "=============================================================="
echo "3) install as root (single docker alpine pass)"
run_ssh "docker run --rm -v ${REMOTE_DIR}:/proj alpine sh -c 'set -e; rm -rf /proj/config/.deploy_extract; mkdir -p /proj/config/.deploy_extract; tar -xzf /proj/config/.deploy_stage/deploy.tar.gz -C /proj/config/.deploy_extract; sh /proj/config/.deploy_extract/install.sh /proj; rm -rf /proj/config/.deploy_extract /proj/config/.deploy_stage/deploy.tar.gz'" || {
  echo "ERROR: remote install failed - cleaning staging" >&2
  run_ssh "rm -rf /proj/config/.deploy_extract /proj/config/.deploy_stage/deploy.tar.gz" >/dev/null 2>&1 || true
  exit 1
}

echo "=============================================================="
echo "4) build + start firewatch (background; first build pulls pip deps)"
# Run detached so a flaky ssh cannot kill the long build; poll for the result.
run_ssh "cd ${REMOTE_DIR} && rm -f /tmp/fw_up.log && (nohup docker compose up -d --build firewatch >/tmp/fw_up.log 2>&1 &) && echo 'build started'" || exit 1

echo "   polling firewatch (this can take minutes on first build)..."
for i in $(seq 1 90); do   # up to ~15 min
  sleep 10
  state="$(run_ssh "cd ${REMOTE_DIR} && docker compose ps --format '{{.Service}}|{{.Status}}' 2>/dev/null | grep '^firewatch|' || true" || true)"
  case "$state" in
    *"Up"*)
      echo "   firewatch is Up after ~$((i * 10))s"
      break
      ;;
  esac
  if ! run_ssh "pgrep -f 'docker compose up -d --build firewatch' >/dev/null" >/dev/null 2>&1; then
    echo "   compose up process ended without 'Up' - build log tail:"
    run_ssh "tail -40 /tmp/fw_up.log" || true
    echo "ERROR: firewatch did not come up - see log above" >&2
    exit 1
  fi
done

if [ "$RESTART_FIREWATCH" -eq 1 ]; then
  echo "   restarting firewatch to load updated model/code/config"
  run_ssh "cd ${REMOTE_DIR} && docker compose restart firewatch" || true
fi
run_ssh "cd ${REMOTE_DIR} && docker compose ps firewatch" || true

echo "=============================================================="
echo "5) verify: --check (config + model load)"
run_ssh "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /scripts/firewatch.py --check" || true

echo "=============================================================="
echo "6) verify: --dry-run (one live pass, prints instead of sending)"
run_ssh "cd ${REMOTE_DIR} && docker compose exec -T firewatch python /scripts/firewatch.py --dry-run" || true

echo "=============================================================="
echo "7) recent firewatch logs"
run_ssh "cd ${REMOTE_DIR} && docker compose logs --since 5m firewatch 2>&1 | tail -40" || true

echo "=============================================================="
echo "DEPLOY COMPLETE"
echo "If a model was promoted, flip its status in models/fire/VERSIONS.md and its"
echo "VERSION.json. Confirm the --dry-run output above before trusting alerts."
