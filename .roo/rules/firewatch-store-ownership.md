# firewatch-store-ownership (never open the live evidence DB from the host)

Applies whenever anything reads or writes firewatch's evidence store
(`<DEPLOY>/media/firewatch.db` on the host = `/media/firewatch/firewatch.db`
inside the container), and whenever a fire alert is missing from the portal.

## The failure mode (hit for real on 2026-09-12: ~2 days of evidence lost)

SQLite's WAL sidecars (`firewatch.db-wal`, `firewatch.db-shm`) carry the
ownership of the process that **created** them, and only that uid (or root) can
write them afterwards. firewatch runs as **uid 1000** in its container, so:

1. a host-side read of the live DB — **even a read-only `mode=ro` open**, e.g.
   an ad-hoc `sqlite3`/python query run as `ai` for analysis — still CREATES
   `-wal`/`-shm` owned by that host user;
2. every later firewatch write fails with
   `attempt to write a readonly database`;
3. firewatch keeps sending **Telegram alerts** (alerting must never stop) while
   the evidence row is lost → the fire event NEVER appears in the portal.

It is silent: the DB is perfectly readable, so nothing looks broken.

## Non-negotiable rules

1. **Never open the live evidence DB from the host** (as `ai`, `dr` or root) —
   not even read-only. Query it *through the container*, as the store owner:

   ```bash
   docker exec -i firewatch python3 -c '
   import sqlite3
   c = sqlite3.connect("file:/media/firewatch/firewatch.db?mode=ro", uri=True)
   for row in c.execute("SELECT id, camera, ts_utc, best_score, alerted "
                        "FROM frames ORDER BY id DESC LIMIT 20"):
       print(row)
   '
   ```

   Anything that runs INSIDE the container creates sidecars owned by uid 1000,
   which is exactly who must own them.

2. **A host-side analysis that needs the data should read a COPY**: pull the DB
   out first (`docker exec firewatch cat /media/firewatch/firewatch.db > /tmp/fw.db`,
   optionally with the `-wal`), then query the copy. The copy lives in the host
   user's own tmp dir, so its sidecars are harmless.

3. **Never `chown`/`chmod` the store to a host user** and never let a host
   process write it: firewatch is the single writer.
