#!/usr/bin/env bash
# ============================================================
# deploy_firewatch.sh - BACKWARD-COMPAT SHIM (deprecated).
#
# Superseded by the unified git-based orchestrator
# dev_scripts/deploy_all.sh (2026-09-05). This shim keeps old references
# working by delegating to the firewatch scope of deploy_all.sh.
#
#   ./dev_scripts/deploy_all.sh firewatch
#
# Deploy is now git-based: local commits are pushed to origin
# (https://github.com/tabebqena/mazr3a) and the host pulls --ff-only,
# so firewatch code/config and the ACTIVE model (best.xml/bin/pt/labelmap.txt,
# now git-tracked) all ride git.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/deploy_all.sh" firewatch
