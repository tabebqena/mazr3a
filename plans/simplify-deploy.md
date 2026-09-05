# Simplify deploy: remove shims, scopes, and bootstrap from `deploy_all.sh`

**Date:** 2026-09-05

## Goal

1. Delete the deprecated shims `dev_scripts/deploy_config.sh` and
   `dev_scripts/deploy_firewatch.sh`.
2. Remove the `config` / `firewatch` / `bootstrap` subcommands from
   `dev_scripts/deploy_all.sh`.
3. The deploy process ALWAYS runs every step on the host (configs + firewatch +
   model) — no scope selection.
4. The user bootstraps the host themselves, so `deploy_all.sh` no longer needs
   the `bootstrap` option (or auto-bootstrap when `.git` is missing).

## Changes

### dev_scripts (deleted)

- `dev_scripts/deploy_config.sh`
- `dev_scripts/deploy_firewatch.sh`

### [`dev_scripts/deploy_all.sh`](../dev_scripts/deploy_all.sh) (rewritten)

- Removed `CMD` / `SCOPE` parsing — the script accepts **no** subcommand and
  always runs the full deploy.
- Removed `bootstrap_host()`.
- Removed the auto-bootstrap branch in `deploy()` (when the host is not yet a
  clone). Now a preflight **fails with instructions** telling the user the host
  must already be bootstrapped (done by hand).
- Removed `BOOTSTRAPPED` first-run special-casing and the `in_scope` helper.
- Config (compose / frigate / mqtt) and firewatch (build / restart / verify)
  steps now run unconditionally, driven only by the git changed-set (an
  unchanged deploy still restarts nothing).
- Header + footer updated to describe the no-subcommand full-deploy flow.

### Docs / helpers updated to match (no `firewatch` subcommand anymore)

- `README.md` — tree dropped the two shims; git-deploy section now says the
  full deploy runs every step and the host clone bootstrap is done by hand.
- `models/fire/VERSIONS.md` — `deploy_all.sh firewatch` → `deploy_all.sh`.
- `models/fire/README.md` — same.
- `dev_scripts/prep_fire_model.sh` — same (comment + hint text).
- `dev_scripts/promote_fire_model.sh` — same (header comment + hint text).

## Notes / assumptions

- The ACTIVE fire model (`models/fire/best.xml/bin/pt/labelmap.txt`) stays
  git-tracked and rides `git pull`; nothing about the git transport changes.
- Host bootstrap steps (git init + remote add origin + fetch + hard reset to
  `origin/master`) are no longer automated by this repo.

## Implementation record

- Deleted `dev_scripts/deploy_config.sh`, `dev_scripts/deploy_firewatch.sh`.
- Rewrote `dev_scripts/deploy_all.sh` (no subcommands, no auto-bootstrap,
  preflight requires a pre-cloned host, all deploy+verify steps run on the
  host).
- Updated live docs/helpers: `README.md`, `models/fire/VERSIONS.md`,
  `models/fire/README.md`, `dev_scripts/prep_fire_model.sh`,
  `dev_scripts/promote_fire_model.sh`.
- Noted in `plans/unified-deploy-orchestrator.md` that it is superseded for the
  scope/bootstrap parts.
- Remote host deploy + verification skipped by user decision (2026-09-05).
  When ready: user bootstraps `/home/dr/frigate` as a clone of origin/master,
  then runs `./dev_scripts/deploy_all.sh` (full deploy + verify).
