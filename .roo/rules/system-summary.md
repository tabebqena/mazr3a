# system-summary (keep SYSTEM_SUMMARY.md current)

`SYSTEM_SUMMARY.md` (repo root) is the **single maintained inventory** of the
system: every compose service, its development-file locations, the host crontab,
required host settings, config inventory (tracked vs git-ignored), storage/disk
cleanup wiring, ports/networking and deploy workflow.

## Non-negotiable

**Any change that adds, removes or modifies a service — including its config,
ports, mounts, cron entries or host requirements — MUST update
[`SYSTEM_SUMMARY.md`](../../SYSTEM_SUMMARY.md) in the SAME commit.** Do not
defer it to a later task.

## What to update when you touch a service

Follow the checklist in the file itself ([§11 Maintaining this file](../../SYSTEM_SUMMARY.md)).
At minimum, update every section that the change touches:

1. **§3 Services** — add/edit the service subsection **and** the
   "Service → development-file map" table.
2. **§4 Host scripts** — if a script under `scripts/` is added/retired.
3. **§5 Crontab** — if a cron entry is added/removed (mirror it in
   `scripts/crontab.sample` or `scripts/crontab.root.sample`).
4. **§6 Required host settings** — new apt package, device, mount or daemon.
5. **§7 Configuration inventory** — new tracked config vs new git-ignored secret.
6. **§8 Storage & cleanup** — add a `config/stores/<service>.conf` profile if the
   service writes files locally.
7. **§9 Ports** — new published or internal port.
8. **§10 Deploy** — if `dev_scripts/deploy_all.sh` steps/verification change.
9. Bump the **Summary version / Last updated** in the header table.

## Portal changes

A portal change also requires bumping `APP_VERSION` in
[`portal/app.py`](../../portal/app.py) and following
[`portal-cache-busting.md`](portal-cache-busting.md).

## Definition of done

- [ ] `SYSTEM_SUMMARY.md` reflects the change.
- [ ] Header "Last updated" (and summary version) bumped.
- [ ] Change committed locally; user is asked to push + deploy
      (`git push origin master` → `./dev_scripts/deploy_all.sh`).
