# system-summary (keep SYSTEM_SUMMARY.md current)

`SYSTEM_SUMMARY.md` (repo root) is the **single maintained inventory** of the
system: every compose service, its development-file locations, the host crontab,
required host settings, config inventory (tracked vs git-ignored), storage/disk
cleanup wiring, ports/networking and deploy workflow.

## Non-negotiable

SYSTEM_SUMMARY.md is a top-level overview, not a changelog. Update it in the same commit ONLY when a change alters the structure of the system (services, images, ports, mounts, volumes, cron entries, host requirements, config inventory, store profiles, deploy steps). Keep cells to one or two lines; per-change detail, dated notes and incident history belong in the relevant plans/*.md doc and the code. Bump the summary version / Last updated; bump APP_VERSION for portal changes.


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
