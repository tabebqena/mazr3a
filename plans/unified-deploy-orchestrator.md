# Unified deploy orchestrator — `dev_scripts/deploy_all.sh` (git-based)

## Goal

Replace the two separate deploy scripts (`dev_scripts/deploy_config.sh` +
`dev_scripts/deploy_firewatch.sh`) with **one comprehensive git-based deploy
orchestrator**:

1. Sync the host machine with the recent local changes (git push + `git pull`).
2. Apply the changes (git pull, fast-forward only) and confirm the host status.
3. Deploy updated configs if any.
4. Deploy the firewatch model (now tracked in git) and run any build steps.
5. Restart frigate.
6. Verify the new configs loaded.
7. Confirm the new model.
8. Confirm the new scripts.

> **Design decision (user, 2026-09-05):** stop hand-rolling file sync. Git
> already solves move/delete/rename natively, so we use a **remote git repo**
> and the deploy command only SSHes to the machine and runs `git pull`.
> The earlier `version_manifest` + md5-free diff design is **superseded and
> removed**.

## Remote git repo

- URL: `https://github.com/tabebqena/mazr3a`
- Branch: `master`
- Local repo currently has **no `origin`** → the plan adds it
  (`git remote add origin https://github.com/tabebqena/mazr3a`) and pushes
  `master`.
- The host clone `/home/dr/frigate` will track this origin and
  `git pull --ff-only origin master`.

## Host bootstrap (one-time)

`/home/dr/frigate` is **NOT a git clone yet** and is the live deploy dir
(owner `dr`; `ai` is in group `dr`, so it can write). `deploy_all.sh bootstrap`
must turn it into a clone **without disturbing git-ignored host state**:

- `.env` (git-ignored secrets)
- `config/telegram.conf` (git-ignored)
- `media/` (git-ignored)
- `mosquitto/data/`, `mosquitto/log/` (git-ignored)
- `models/fire/versions/` (git-ignored archive)

Bootstrap steps (idempotent):

1. `git -C /home/dr/frigate init` if no `.git`.
2. `git -C /home/dr/frigate remote add origin https://github.com/tabebqena/mazr3a`
   (or `set-url` if present).
3. `git -C /home/dr/frigate fetch origin`
4. `git -C /home/dr/frigate checkout -B master origin/master` (keeps ignored files)
   — never touches `.env`/`telegram.conf`/`media`/`versions/`.

## ACTIVE fire model now tracked

The firewatch ACTIVE OpenVINO IR (`models/fire/best.xml`, `best.bin`,
`labelmap.txt`) and the source checkpoint (`best.pt`) are **committed to git**
so the whole deploy including the model rides `git pull`. `.gitignore` change:

- Stop ignoring `models/fire/*.xml`, `models/fire/*.bin`, `models/fire/*.pt`,
  `models/fire/labelmap.txt`.
- Keep ignoring `models/fire/versions/` (versioned archive stays local-only,
  documented in `models/fire/VERSIONS.md`).

## Pipeline (deploy_all.sh, default scope = all)

```
deploy_all.sh                # full deploy
deploy_all.sh config         # frigate/mqtt config deploy only
deploy_all.sh firewatch      # firewatch code/config/model deploy only
deploy_all.sh bootstrap      # one-time: turn /home/dr/frigate into a clone
```

Default flow:

0. Preflight: local git repo clean enough to push, ssh reachable, host has the
   remote (bootstrap if `.git` absent).
1. Sync: `git push origin master` (local) then
   `ssh ai@ssh.mazr3a.garden "cd /home/dr/frigate && git pull --ff-only origin master"`.
2. Determine what changed between the host's previous HEAD and new HEAD
   (`git diff --name-only <old>..<new>`) — this replaces any file-diff/md5 step.
3. Restart services based on the changed set (scope-aware):
   - `docker-compose.yml` changed → `docker compose up -d`
   - `config/config.yaml` changed → restart frigate
   - `mosquitto/config/mosquitto.conf` changed → restart mqtt
   - `firewatch/*` / `scripts/firewatch.py` / `config/firewatch.conf` /
     `models/fire/*` changed → restart firewatch (build if `Dockerfile`/
     `requirements.txt` changed)
4. Verify:
   - config loaded: `docker compose ps` + `verify_remote.py` + log scan
   - model: `firewatch.py --check` (loads IR) + confirm tracked model files on
     host are at the pulled commit (`git status` clean / `git rev-parse`)
   - scripts: pulled by git (confirmed by HEAD) + `firewatch.py --dry-run`

## File/ownership notes

- The two old scripts become thin shims:
  - `deploy_config.sh` → `exec deploy_all.sh config`
  - `deploy_firewatch.sh` → `exec deploy_all.sh firewatch`
- `version_manifest` and `dev_scripts/bump_manifest.sh` (created during the
  earlier file-diff design) are **deleted** — git is the source of truth.
- `.roo/rules/Agents.md` already requires a git commit after each change; the
  deploy flow relies on that (commit → push → pull). No extra rule needed.

## Implementation record

- **Plan written** (this file), then executed:
- Rewrote `dev_scripts/deploy_all.sh` — git-based: `push` local `master` →
  ensure host `/home/dr/frigate` is a clone (auto-`bootstrap` if not) →
  capture host HEAD → `git pull --ff-only` → diff old..new HEAD → restart
  only the services whose files changed → verify configs/models/scripts.
  Commands: `deploy` (all), `config`, `firewatch`, `bootstrap`. Transport
  keeps the SSH_ASKPASS + retry pattern for the flaky cloudflared tunnel.
- Removed `version_manifest` and `dev_scripts/bump_manifest.sh` (the earlier
  file-diff/md5-free design is superseded by git).
- `.gitignore` — ACTIVE fire model (`models/fire/best.xml/bin/pt/labelmap.txt`)
  is now tracked; `models/fire/versions/` + secrets stay ignored.
- `deploy_config.sh` / `deploy_firewatch.sh` → thin shims delegating to
  `deploy_all.sh config` / `deploy_all.sh firewatch`.
- Docs updated: `README.md` (git deploy section + tree), `models/fire/README.md`,
  `models/fire/VERSIONS.md`, plus `prep_fire_model.sh`/`promote_fire_model.sh`
  hints (commit ACTIVE set + `deploy_all.sh firewatch`).
- `origin` set to `https://github.com/tabebqena/mazr3a`; `master` not pushed
  yet (needs GitHub credentials - done on the real dev machine).
- Verified: `bash -n` on all edited shell scripts; review done.
- Commit created per .roo/rules/Agents.md.
- Deploy to the remote host is pending per .roo/rules/sshuser.md: ask + run
  `./dev_scripts/deploy_all.sh bootstrap` once, then `deploy` + verify there.
