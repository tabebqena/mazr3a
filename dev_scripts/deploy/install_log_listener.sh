#!/usr/bin/env bash
# Install (or remove) a systemd service that runs the general port listener on
# port 514 — WITHOUT running it as root and WITHOUT a global `setcap` on
# python. The unit grants only CAP_NET_BIND_SERVICE via systemd.
#
# Port 514 is privileged (<1024): binding it as a normal user fails with
# "Permission denied". This installer is the clean fix.
#
# Usage:
#   sudo ./dev_scripts/deploy/install_log_listener.sh            # install + start
#   sudo ./dev_scripts/deploy/install_log_listener.sh --remove   # stop + uninstall
#
# After install:
#   systemctl status nvr-log-listener
#   journalctl -u nvr-log-listener -f
#   tail -f /var/log/nvr/capture.log
#   # allow the NVR to reach it:
#   sudo ufw allow 514/udp && sudo ufw allow 514/tcp   # if ufw is in use
#
# NOTE: This script changes the system (systemd unit + /var/log/nvr) and must be
# run by the operator with sudo. The AI does not run it.
set -euo pipefail

UNIT=/etc/systemd/system/nvr-log-listener.service
OUTDIR="${NVR_LOG_OUTDIR:-/var/log/nvr}"
PORT="${NVR_LOG_PORT:-514}"

if [[ "${1:-}" == "--remove" || "${1:-}" == "uninstall" ]]; then
  [[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
  systemctl disable --now nvr-log-listener 2>/dev/null || true
  rm -f "$UNIT"
  systemctl daemon-reload
  echo "removed $UNIT (kept $OUTDIR data)"
  exit 0
fi

[[ $EUID -eq 0 ]] || { echo "run with sudo: sudo $0" >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LISTENER="$REPO_DIR/dev_scripts/deploy/port_listener.py"
[[ -f "$LISTENER" ]] || { echo "listener not found: $LISTENER" >&2; exit 1; }

PYTHON="$(command -v python3)"
RUN_USER="${SUDO_USER:-$(id -un)}"
[[ "$RUN_USER" == "root" ]] && echo "warn: SUDO_USER unset; running service as root" >&2

install -d -o "$RUN_USER" -g "$RUN_USER" -m 0755 "$OUTDIR"

cat > "$UNIT" <<EOF
[Unit]
Description=NVR log listener (general TCP+UDP :$PORT)
Documentation=file:$LISTENER
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$(id -gn "$RUN_USER")
WorkingDirectory=$REPO_DIR
ExecStart=$PYTHON $LISTENER --port $PORT --proto both --outdir $OUTDIR
Restart=always
RestartSec=3

# Allow binding the privileged port 514 without running as root:
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=yes

# Sandboxing (listener needs write access only to its outdir):
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=yes
ReadWritePaths=$OUTDIR

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now nvr-log-listener
sleep 1
systemctl --no-pager --full status nvr-log-listener || true

echo
echo "== installed =="
echo "unit    : $UNIT"
echo "out dir : $OUTDIR  (raw: capture.raw, index: records.jsonl, log: capture.log)"
echo "logs    : journalctl -u nvr-log-listener -f"
echo "firewall: sudo ufw allow $PORT/udp && sudo ufw allow $PORT/tcp   (if ufw)"
echo "next    : point the NVR Log Server at <this-host-ip>:$PORT"
