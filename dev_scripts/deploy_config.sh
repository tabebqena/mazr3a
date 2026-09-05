#!/usr/bin/env bash
# ============================================================
# deploy_config.sh - BACKWARD-COMPAT SHIM (deprecated).
#
# Superseded by the unified git-based orchestrator
# dev_scripts/deploy_all.sh (2026-09-05). This shim keeps old references
# working by delegating to the config scope of deploy_all.sh.
#
#   ./dev_scripts/deploy_all.sh config
#
# Deploy is now git-based: local commits are pushed to origin
# (https://github.com/tabebqena/mazr3a) and the host pulls --ff-only,
# so config.yaml + all tracked files ride git.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/deploy_all.sh" config
