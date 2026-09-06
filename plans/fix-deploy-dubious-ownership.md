# Fix deploy_all.sh "dubious ownership" — configurable SSH credentials, no safe.directory

**Date:** 2026-09-06

## Goal

`deploy_all.sh` currently hard-codes the SSH credentials (`ai` / `123456`) and the
tunnel host. When it runs git on the host clone at `/home/dr/frigate` it fails with:

    fatal: detected dubious ownership in repository at '/home/dr/frigate'

because the clone is owned by `dr`, the deploy SSHes in as `ai`, and git ≥ 2.35.2
refuses to operate on a repo owned by a different user (group write access is NOT
enough). The `run_ssh()` retry loop then re-tries the same NON-transient failure
5× before giving up.

Decisions from the user:
1. **Do NOT** mark `/home/dr/frigate` as a git `safe.directory` (user direction).
2. Instead, make the SSH credentials **configurable** so the deploy can run as the
   user who OWNS the host clone (`dr`) — git's ownership check then passes and no
   safe.directory is needed.
3. Credentials are read from the **environment** first
   (`DEPLOY_SSH_USER` / `DEPLOY_SSH_PASS`, plain `SSH_USER` / `SSH_PASSWORD` are
   also honoured) and **prompted interactively** when not set. Nothing is
   hard-coded anymore.

## Changes

### [`dev_scripts/deploy_all.sh`](../dev_scripts/deploy_all.sh)

- Remove the hard-coded `SSH_USER="ai"` and `SSH_PASS="123456"`.
- Read credentials from env (with prompt fallback):
  - `SSH_USER="${DEPLOY_SSH_USER:-${SSH_USER:-}}"`
  - `SSH_PASS="${DEPLOY_SSH_PASS:-${SSH_PASSWORD:-}}"`
- New `resolve_ssh_credentials()`: if either value is empty and stdin is a TTY,
  ask for it (`read -r -s` for the password); if stdin is not a TTY, fail with a
  clear message telling the user to export `DEPLOY_SSH_USER` / `DEPLOY_SSH_PASS`.
- Make the SSH_ASKPASS helper robust: it now `cat`s a temp password file (chmod
  600) instead of `echo`ing the password inline, so a user-typed password with
  spaces/quotes/shell metacharacters is passed to ssh verbatim.
- Allow `DEPLOY_SSH_HOST` to override the tunnel host (default unchanged).
- Update the header TRANSPORT comment: configurable credentials; the SSH user
  must OWN the host clone (git ≥ 2.35.2 "dubious ownership"; deliberately no
  `safe.directory`); on this host the owner is `dr`.

## Notes / assumptions

- The deploy runs every step (git + docker) over one SSH session as the chosen
  user. On the current host `/home/dr/frigate` is owned by `dr`, so run with
  `DEPLOY_SSH_USER=dr` (or answer the prompt with `dr`) — git then works with no
  `safe.directory`. If the clone were re-owned to another account later, connect
  as that owner instead.
- Non-interactive/automated use (`DEPLOY_ASSUME_YES=1`, cron, CI) must export
  `DEPLOY_SSH_USER` and `DEPLOY_SSH_PASS`.
- The tunnel password itself is unchanged; only where it comes from changes.

## Implementation record

- Removed hard-coded `SSH_USER="ai"` / `SSH_PASS="123456"` from
  [`dev_scripts/deploy_all.sh`](../dev_scripts/deploy_all.sh).
- Credentials now come from env (`DEPLOY_SSH_USER` / `DEPLOY_SSH_PASS`, plain
  `SSH_USER` / `SSH_PASSWORD` also honoured) or are prompted interactively;
  added `resolve_ssh_credentials()` with a clear non-TTY error.
- `DEPLOY_SSH_HOST` override added (default `ssh.mazr3a.garden` unchanged).
- SSH_ASKPASS now cats a chmod-600 password file (verbatim password, safe for
  spaces/quotes) instead of inlining an `echo`; trap cleans both temp files.
- Header TRANSPORT comment rewritten: configurable creds + run as the clone
  owner (`dr`) because git >= 2.35.2 "dubious ownership" and no `safe.directory`
  is added (user decision).
- Verified locally: `bash -n` clean; env resolution keeps passwords verbatim;
  no-credentials non-TTY run aborts (exit 1) before any SSH.
- Remote host verification pending (user pushes to origin + re-runs deploy).
