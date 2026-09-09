#!/usr/bin/env bash
# ============================================================
# cleanup_media.sh - SUPERSEDED (2026-09-09) - DO NOT SCHEDULE.
# Retired in favour of the unified disk heartbeat:
#   scripts/heartbeat_cleanup.py + config/heartbeat.conf +
#   config/stores/frigate.conf (config/stores/*.conf per service).
# Kept on disk only as a reference / manual one-shot fallback.
# ============================================================
# ============================================================
# (Original description preserved below for reference.)
# cleanup_media.sh - Delete the oldest Frigate media files when
# the media directory grows past a configurable disk cap (or free
# space drops below a minimum).
#
# Tunables live in config/cleanup_media.conf (same layout as the
# deployed ~/frigate tree). Override the path with CLEANUP_CONF.
#
# Triggers (each can be disabled with 0 in the config):
#   MAX_MEDIA_GB  -> delete oldest-first while usage is over the cap
#   MIN_FREE_GB   -> delete oldest-first while free space is low
#   MAX_AGE_DAYS  -> additionally delete files older than N days
#
# Intended to run from the host crontab (see scripts/crontab.sample).
# ============================================================

set -u   # treat unset variables as errors (but no set -e: we handle
         # each step explicitly so one failure cannot kill a run)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${CLEANUP_CONF:-${SCRIPT_DIR}/../config/cleanup_media.conf}"

# --- Defaults (overridden by the config file) ---------------------
MEDIA_DIR=""
MAX_MEDIA_GB=0
MIN_FREE_GB=0
MAX_AGE_DAYS=0
LOG_FILE="/tmp/media-cleanup.log"

if [[ -f "$CONF_FILE" ]]; then
  # shellcheck source=/dev/null
  . "$CONF_FILE"
else
  echo "[cleanup] WARNING: config not found at $CONF_FILE - using defaults"
fi

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

# --- Safety guards -------------------------------------------------
if [[ -z "$MEDIA_DIR" ]]; then
  echo "[cleanup] ERROR: MEDIA_DIR is empty - refusing to run" | tee -a "$LOG_FILE"
  exit 1
fi

MEDIA_DIR="$(cd "$MEDIA_DIR" 2>/dev/null && pwd)" || {
  echo "[cleanup] ERROR: MEDIA_DIR does not exist: $MEDIA_DIR" | tee -a "$LOG_FILE"
  exit 1
}

case "$MEDIA_DIR" in
  /|/home|/root|/var|/etc|/usr|/boot|/opt)
    echo "[cleanup] ERROR: refusing to operate on protected dir: $MEDIA_DIR" | tee -a "$LOG_FILE"
    exit 1
    ;;
esac

# --- Convert caps to KB -------------------------------------------
MAX_MEDIA_KB=$((MAX_MEDIA_GB * 1024 * 1024))
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))

# --- Measure current state ----------------------------------------
usage_kb=$(du -sk "$MEDIA_DIR" 2>/dev/null | awk '{print $1}')
free_kb=$(df -kP "$MEDIA_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
usage_kb=${usage_kb:-0}
free_kb=${free_kb:-0}

log "usage=${usage_kb}KB cap=${MAX_MEDIA_KB}KB free=${free_kb}KB minfree=${MIN_FREE_KB}KB"

needs_cleanup() {
  local over_cap=0 low_space=0
  if [[ "$MAX_MEDIA_GB" -gt 0 && "$usage_kb" -gt "$MAX_MEDIA_KB" ]]; then
    over_cap=1
  fi
  if [[ "$MIN_FREE_GB" -gt 0 && "$free_kb" -lt "$MIN_FREE_KB" ]]; then
    low_space=1
  fi
  [[ "$over_cap" -eq 1 || "$low_space" -eq 1 ]]
}

recheck() {
  usage_kb=$(du -sk "$MEDIA_DIR" 2>/dev/null | awk '{print $1}')
  free_kb=$(df -kP "$MEDIA_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
  usage_kb=${usage_kb:-0}
  free_kb=${free_kb:-0}
}

# --- Cap-driven cleanup: delete oldest-first ----------------------
deleted=0
if needs_cleanup; then
  log "over cap / low space - deleting oldest files first"
  now_epoch=$(date +%s)
  batch=0
  while IFS= read -r -d '' entry; do
    # entry format: <epoch>\t<path>
    epoch="${entry%%$'\t'*}"
    file="${entry#*$'\t'}"

    # re-check every 25 deletions so we don't over-delete
    if (( batch > 0 && batch % 25 == 0 )) && ! needs_cleanup; then
      break
    fi

    # skip if MAX_AGE_DAYS is set and this file is NEWER than the limit
    if [[ "$MAX_AGE_DAYS" -gt 0 ]]; then
      age_sec=$((now_epoch - epoch))
      if (( age_sec < MAX_AGE_DAYS * 86400 )); then
        batch=$((batch + 1))
        continue
      fi
    fi

    if rm -f -- "$file" 2>/dev/null; then
      deleted=$((deleted + 1))
    fi
    batch=$((batch + 1))
    recheck
  done < <(find "$MEDIA_DIR" -type f -printf '%T@\t%p\0' | sort -nz)
fi

# --- Age-based sweep (independent safety net) ----------------------
age_deleted=0
if [[ "$MAX_AGE_DAYS" -gt 0 ]]; then
  while IFS= read -r -d '' file; do
    if [[ -f "$file" ]]; then
      rm -f -- "$file" 2>/dev/null && age_deleted=$((age_deleted + 1))
    fi
  done < <(find "$MEDIA_DIR" -type f -mtime +"$((MAX_AGE_DAYS - 1))" -print0)
fi

# --- Remove now-empty subdirectories (recording day folders etc.) --
# -mindepth 1 keeps the media ROOT itself intact even if it is empty.
find "$MEDIA_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null

recheck
log "done: cap_deleted=${deleted} age_deleted=${age_deleted} usage=${usage_kb}KB free=${free_kb}KB"
exit 0
