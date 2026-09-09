# ssh-password (no sshpass — use OpenSSH SSH_ASKPASS)

Applies whenever the AI must SSH/SCP **non-interactively** to the remote host
(`ssh.mazr3a.garden`), e.g. remote verification, smoke tests, or running
remote helpers. The canonical implementation lives in
[`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh:101) (`run_ssh()`, the
SSH_ASKPASS setup at lines 101–124). Copy/adapt that — do not re-invent and do
not fall back to `sshpass`.

## Non-negotiable

1. **`sshpass` is NOT installed on this dev machine and is NEVER to be used,
   installed, or defaulted to.** Do not emit `sshpass -p … ssh` / `scp`, do not
   `apt/brew install sshpass`, and do not add it to any Dockerfile, script, or
   plan. There is always an SSH_ASKPASS-based way (below).
2. **Never hard-code credentials** in scripts, plans, or this rule. SSH
   credentials come from the environment
   (`DEPLOY_SSH_USER` / `DEPLOY_SSH_PASS`; plain `SSH_USER` / `SSH_PASSWORD` are
   also honoured) or are prompted interactively. The `ai` / `123456` pair
   documented in [`.roo/rules/sshuser.md`](.roo/rules/sshuser.md) is for the
   **user's own interactive sessions** — not for the AI to embed in commands.

## Strategy — SSH_ASKPASS (OpenSSH built-in, no external tool)

For non-interactive, password-based SSH/SCP without a TTY:

1. **Write the password to a temp file (verbatim, no shell re-interpretation):**
   ```bash
   PASSFILE="$(mktemp)"
   printf '%s\n' "$SSH_PASS" > "$PASSFILE"
   chmod 600 "$PASSFILE"
   ```
2. **Create a tiny helper that cats it back to ssh:**
   ```bash
   ASKPASS="$(mktemp)"
   printf '#!/usr/bin/env bash\ncat "%s"\n' "$PASSFILE" > "$ASKPASS"
   chmod 700 "$ASKPASS"
   ```
3. **Make ssh consult it even without a TTY:**
   ```bash
   export SSH_ASKPASS="$ASKPASS"
   export SSH_ASKPASS_REQUIRE=force   # use askpass even with no controlling tty
   export DISPLAY=:0                  # legacy requirement for the askpass path
   ```
4. **Run under `setsid`** so there is no controlling terminal, with
   password-only, single-prompt options:
   ```bash
   setsid /usr/bin/ssh \
     -o StrictHostKeyChecking=accept-new \
     -o PreferredAuthentications=password \
     -o PubkeyAuthentication=no \
     -o NumberOfPasswordPrompts=1 \
     -o ConnectTimeout=25 \
     -o ServerAliveInterval=15 -o ServerAliveCountMax=6 \
     "${SSH_USER}@ssh.mazr3a.garden" "$cmd"
   ```
5. **Never leave the plaintext on disk** — clean up on exit:
   ```bash
   trap 'rm -f "$ASKPASS" "$PASSFILE"' EXIT
   ```

Retry flaky tunnel connections (up to ~5× with a pause), exactly as
`run_ssh()` in [`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh:126) does.

## Ownership boundaries (align with [`.roo/rules/sshuser.md`](.roo/rules/sshuser.md))

- Deploy / git handoff on the host is owned by `deploy_all.sh` running as **`dr`**
  (the owner of `/home/dr/frigate`). The AI must **not** run git on the host as
  `ai` (dubious ownership) and must **not** pull from the host side.
- One-off remote **verification** as `ai` is allowed for **docker/compose checks
  only**. Run it interactively (ask the user to type the password) or, if it must
  be automated, with the SSH_ASKPASS recipe above.
- Fully unattended runs (cron/CI/`DEPLOY_ASSUME_YES=1`) **require**
  `DEPLOY_SSH_USER` / `DEPLOY_SSH_PASS` to be exported; if they are missing and
  stdin is not a TTY, abort with a clear message (see `resolve_ssh_credentials()`
  in [`dev_scripts/deploy_all.sh`](dev_scripts/deploy_all.sh:75)).
