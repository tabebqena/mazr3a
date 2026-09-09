#!/usr/bin/env python3
"""heartbeat_cleanup.py - the unified disk heartbeat: one periodic host cleaner
that keeps the hard disk from filling up across ALL locally-writing services.

Replaces the old pair of cleaners:
  - scripts/cleanup_media.sh      (Frigate media, host bash)  -> RETIRED
  - the firewatch-only cron line  -> now a delegated store below

Every service that stores files locally declares its OWN store profile under
config/stores/<service>.conf (KEY=VALUE, same layout as the other conf files).
Each profile tells the heartbeat how to deal with that service's files:

  TYPE=dir        a plain on-disk store the heartbeat purges itself
                  (Frigate media, watchdog CSVs, mosquitto data/log, host logs)
  TYPE=docker-exec a DB-aware store whose in-container worker owns the files
                  (firewatch evidence; the SQLite DB is the source of truth)

Per-service rules (in each profile):
  MAX_AGE_DAYS    delete files older than N days (always-on belt)
  MAX_SIZE_GB     delete oldest-first while the store is over this cap
  ROOT/SUBDIRS    WHERE the service's files are (scoped - never the whole
                  shared ./media tree, so no store can touch another's files)
  PRIORITY        escalation order when the GLOBAL cap is hit (lower first)

The GLOBAL cap lives in config/heartbeat.conf:
  FS_PATH, MIN_FREE_GB / MAX_USED_PCT (trigger), RELIEF_FREE_GB (stop point)
Two phases run on every tick:
  Phase 1 - per-store compliance (each store enforces its own age/size caps)
  Phase 2 - global-cap escalation (disk over the global cap -> free oldest
            files across stores in PRIORITY order until RELIEF_FREE_GB)

Runs from the host ROOT crontab (see scripts/crontab.root.sample) every
15 min; the dr-level monitor/sampler jobs stay in scripts/crontab.sample.
Root is required for real runs because deleting root-owned service dirs
(Frigate media) needs it; a permission pre-flight verifies every store dir
before any deletion so a missed permission never silently fails.

Modes:
  heartbeat_cleanup.py               # real run (requires root)
  heartbeat_cleanup.py --dry-run     # report only, change nothing
  heartbeat_cleanup.py --check       # per-store permission/usage table, no change

Env overrides: HEARTBEAT_CONF (central conf path), HEARTBEAT_DEPLOY (deploy
root; the {DEPLOY} token in configs expands to it), HEARTBEAT_STORES_DIR.
CLI flags (--deploy/--stores-dir/--log-file/--fs-path) win over env/conf.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

# scripts/ dir on sys.path so telegram_notify (optional) is importable.
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)

_GB = 1024 ** 3

# Directories that must NEVER be a managed store root (mirrors the old
# cleanup_media.sh guard; exact realpath equality is checked).
_PROTECTED_ROOTS = {"/", "/home", "/root", "/var", "/etc", "/usr", "/boot",
                    "/opt", "/proc", "/sys", "/dev", "/run"}

# Basenames NEVER deleted, regardless of a store's FILE_PATTERNS/EXCLUDE_GLOB.
# Live DBs, locks, sockets and state must survive any cleanup.
_HARD_PROTECT = ("*.db", "*.db-wal", "*.db-shm", "*.lock", "*.pid", "*.sock",
                 "*.state")

# Re-measure free space / store usage this often while deleting (keeps the
# delete loops from overshooting the caps).
_RECHECK_EVERY = 25


# ---------------------------------------------------------------------------
# tiny KEY=VALUE conf reader (same style as the rest of the project)
# ---------------------------------------------------------------------------
def read_conf(path):
    """Read a KEY=VALUE file (skip blank + full-line # comments) into a dict."""
    cfg = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    except (FileNotFoundError, OSError):
        return {}
    return cfg


def expand(cfg, deploy):
    """Return a copy of cfg with every {DEPLOY} token replaced by deploy."""
    out = {}
    for key, value in cfg.items():
        out[key] = value.replace("{DEPLOY}", deploy) if isinstance(value, str) \
            else value
    return out


def _int(value, default=0):
    try:
        return int(str(value).strip() or default)
    except (TypeError, ValueError):
        return default


def _float(value, default=0.0):
    try:
        return float(str(value).strip() or default)
    except (TypeError, ValueError):
        return default


def _bool(value, default=False):
    return str(value).strip().lower() in ("1", "true", "yes", "on") \
        if value is not None else default


def _list(value):
    """Split a comma-separated config value into stripped non-empty items."""
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def __call__(self, msg):
        line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
        print(line, flush=True)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass  # logging is best-effort


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def fmt_gb(n):
    return f"{n:.1f}GiB"


# ---------------------------------------------------------------------------
# filesystem helpers
# ---------------------------------------------------------------------------
def fs_usage(path):
    """shutil.disk_usage(path) -> (total, used, free) bytes; None on failure."""
    try:
        return shutil.disk_usage(path)
    except OSError:
        return None


def du_bytes(paths):
    """On-disk usage (bytes) of the given dirs via `du -sk`. 0 on failure."""
    if not paths:
        return 0
    try:
        result = subprocess.run(
            ["du", "-sk"] + list(paths), capture_output=True, text=True,
            timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    total_kb = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if parts:
            try:
                total_kb += int(parts[0])
            except ValueError:
                pass
    return total_kb * 1024


def iter_entries(root, max_depth, follow_symlinks=False):
    """Yield (path, is_dir) under root; max_depth None = unlimited.

    max_depth is how many directory LEVELS below root are traversed
    (0 = only files/dirs directly inside root). Symlinked dirs are not
    followed unless follow_symlinks.
    """
    def rec(current, level):
        if max_depth is not None and level > max_depth:
            return
        try:
            entries = list(os.scandir(current))
        except OSError:
            return
        for entry in entries:
            try:
                is_dir = entry.is_dir(follow_symlinks=follow_symlinks)
                is_link = entry.is_symlink()
            except OSError:
                continue
            if is_dir:
                yield entry.path, True
                if not is_link:
                    rec(entry.path, level + 1)
            else:
                yield entry.path, False
    yield from rec(root, 0)


def protected(path):
    """True if realpath(path) is exactly one of the protected roots."""
    try:
        real = os.path.realpath(path)
    except OSError:
        return True
    return real in _PROTECTED_ROOTS


def hard_protected(basename):
    import fnmatch
    return any(fnmatch.fnmatch(basename, pat) for pat in _HARD_PROTECT)


def is_writable_dir(path):
    """Can we create/delete entries inside this dir? (unlink needs W|X on it.)"""
    return os.access(path, os.W_OK | os.X_OK)


# ---------------------------------------------------------------------------
# store profiles
# ---------------------------------------------------------------------------
def load_stores(stores_dir, deploy):
    """Load every <service>.conf under stores_dir into a dict keyed by name."""
    stores = {}
    if not os.path.isdir(stores_dir):
        return stores
    for name in sorted(os.listdir(stores_dir)):
        if not name.endswith(".conf"):
            continue
        raw = expand(read_conf(os.path.join(stores_dir, name)), deploy)
        if not _bool(raw.get("ENABLED"), True):
            continue
        service = os.path.basename(name)[:-len(".conf")]
        stores[service] = {
            "file": os.path.join(stores_dir, name),
            "name": raw.get("NAME") or service,
            "service": service,
            "type": raw.get("TYPE", "dir").strip().lower(),
            "root": raw.get("ROOT", "").strip(),
            "subdirs": _list(raw.get("SUBDIRS")),
            "depth": None if not str(raw.get("DEPTH", "")).strip()
                     else max(0, _int(raw.get("DEPTH"), 0)),
            "file_patterns": _list(raw.get("FILE_PATTERNS")),
            "exclude_glob": _list(raw.get("EXCLUDE_GLOB")),
            "max_age_days": _float(raw.get("MAX_AGE_DAYS"), 0.0),
            "max_size_gb": _float(raw.get("MAX_SIZE_GB"), 0.0),
            "priority": _int(raw.get("PRIORITY"), 50),
            "exec_cmd": raw.get("EXEC_CMD", "").strip(),
            "exec_dry_flag": raw.get("EXEC_DRY_FLAG", "").strip(),
            "exec_args": _list(raw.get("EXEC_ARGS")),
        }
    return stores


def managed_dirs(store):
    """List of absolute dirs the store is allowed to manage (expanded)."""
    root = store["root"]
    if not root:
        return []
    if store["subdirs"]:
        return [os.path.join(root, sub) for sub in store["subdirs"]]
    return [root]


# ---------------------------------------------------------------------------
# candidate collection (eligible deletable files, oldest first)
# ---------------------------------------------------------------------------
def collect_candidates(store, log, require_writable=True):
    """Return (files, total_bytes) where files is a list of dicts:
    {path, mtime, size} for every deletable file under the store's managed
    dirs, sorted oldest-first. Honours DEPTH, FILE_PATTERNS, EXCLUDE_GLOB,
    HARD_PROTECT and never follows symlinks out of the store.

    dirs that are missing are logged and skipped; an existing-but-unwritable
    dir is an error (skipped) unless require_writable is False (check mode).
    """
    import fnmatch
    files = []
    total = 0
    seen_dirs = set()

    def eligible(basename):
        if hard_protected(basename):
            return False
        if store["exclude_glob"] and any(
                fnmatch.fnmatch(basename, pat)
                for pat in store["exclude_glob"]):
            return False
        if store["file_patterns"] and not any(
                fnmatch.fnmatch(basename, pat)
                for pat in store["file_patterns"]):
            return False
        return True

    for d in managed_dirs(store):
        real = os.path.realpath(d)
        if real in seen_dirs:
            continue
        seen_dirs.add(real)
        if not os.path.isdir(d):
            log(f"  store {store['name']}: missing dir {d} - skipped")
            continue
        if protected(d):
            log(f"  store {store['name']}: REFUSING protected dir {d}")
            continue
        if require_writable and not is_writable_dir(d):
            log(f"  store {store['name']}: NOT writable by euid "
                f"{os.geteuid()} - {d} skipped (permission pre-flight)")
            continue
        for path, is_dir in iter_entries(d, store["depth"]):
            if is_dir:
                continue
            base = os.path.basename(path)
            if not eligible(base):
                continue
            try:
                st = os.stat(path)  # follows symlink target for size
                is_link = os.path.islink(path)
            except OSError:
                continue
            # never delete a symlink whose target escapes the store tree
            if is_link:
                try:
                    target_real = os.path.realpath(path)
                except OSError:
                    continue
                if not any(
                        target_real == os.path.realpath(md) or
                        target_real.startswith(os.path.realpath(md) + os.sep)
                        for md in managed_dirs(store)):
                    continue
            if not os.path.isfile(path):
                continue
            files.append({"path": path, "mtime": st.st_mtime,
                          "size": st.st_size})
            total += st.st_size
    files.sort(key=lambda f: f["mtime"])  # oldest first
    return files, total


# ---------------------------------------------------------------------------
# deletion core
# ---------------------------------------------------------------------------
def delete_oldest(store, files, log, dry_run, reason, max_bytes=None,
                  max_count=None, cutoff=None):
    """Delete files (oldest-first) while constraints allow.

    Files is a sorted-oldest list of {path,size,mtime}. Returns
    (deleted_count, freed_bytes). Constraints (all optional):
      max_bytes - stop once freed >= max_bytes
      max_count - stop once deleted >= max_count
      cutoff    - only delete files with mtime < cutoff
      reason    - label used in dry-run/log lines
    """
    deleted = 0
    freed = 0
    for f in files:
        if max_count is not None and deleted >= max_count:
            break
        if max_bytes is not None and freed >= max_bytes:
            break
        if cutoff is not None and f["mtime"] >= cutoff:
            continue
        if dry_run:
            log(f"  dry: [{reason}] {f['path']} "
                f"({fmt_bytes(f['size'])})")
        else:
            try:
                os.unlink(f["path"])
            except OSError as exc:
                log(f"  {store['name']}: delete failed {f['path']}: {exc}")
                continue
        deleted += 1
        freed += f["size"]
    return deleted, freed


def remove_empty_dirs(store, log, dry_run):
    """Best-effort: rmdir now-empty subdirs strictly below each managed dir.

    Never removes the managed dir itself and never follows symlinks, so it
    can only remove dirs inside the store's own scoped tree.
    """
    removed = 0
    for d in managed_dirs(store):
        if not os.path.isdir(d) or protected(d):
            continue
        for root, dirnames, _filenames in os.walk(d, topdown=False):
            for dn in dirnames:
                full = os.path.join(root, dn)
                if os.path.islink(full):
                    continue
                try:
                    os.rmdir(full)
                    removed += 1
                    if dry_run:
                        log(f"  dry: [empty-dir] {full}")
                except OSError:
                    pass
    if removed and not dry_run:
        log(f"  store {store['name']}: removed {removed} empty dir(s)")
    return removed


# ---------------------------------------------------------------------------
# phase 1: per-store compliance (age + size caps)
# ---------------------------------------------------------------------------
def phase1_store(store, log, dry_run):
    if store["type"] == "docker-exec":
        return run_worker(store, log, dry_run)

    cap_bytes = store["max_size_gb"] * _GB
    age_cutoff = time.time() - store["max_age_days"] * 86400 \
        if store["max_age_days"] > 0 else None
    if cap_bytes <= 0 and age_cutoff is None:
        log(f"  store {store['name']}: no age/size cap - compliance skip")
        return 0, 0

    files, total = collect_candidates(store, log, require_writable=True)
    log(f"  store {store['name']}: candidates={len(files)} "
        f"store_bytes={fmt_bytes(total)} "
        f"age_cap={store['max_age_days']:g}d "
        f"size_cap={fmt_bytes(cap_bytes) if cap_bytes > 0 else 'off'}")
    total_del = 0
    total_freed = 0

    # age belt: delete every eligible file older than the age cap
    if age_cutoff is not None and files:
        n, freed = delete_oldest(store, files, log, dry_run,
                                 reason=f"age>{store['max_age_days']:g}d",
                                 cutoff=age_cutoff)
        total_del += n
        total_freed += freed
        files = [f for f in files if f["mtime"] >= age_cutoff]

    # size cap: while the store's real on-disk usage is over cap, delete the
    # oldest remaining eligible files. Re-measure with du every _RECHECK_EVERY
    # deletions (like the old cleanup_media.sh) so we don't over-delete.
    if cap_bytes > 0:
        n = 0
        freed = 0
        usage_now = du_bytes(managed_dirs(store))
        while files and usage_now > cap_bytes:
            f = files.pop(0)
            if dry_run:
                log(f"  dry: [size>{store['max_size_gb']:g}G] {f['path']} "
                    f"({fmt_bytes(f['size'])})")
                n += 1
                freed += f["size"]
                usage_now -= f["size"]
                continue
            try:
                os.unlink(f["path"])
            except OSError as exc:
                log(f"  {store['name']}: delete failed {f['path']}: {exc}")
                continue
            n += 1
            freed += f["size"]
            usage_now -= f["size"]
            if n % _RECHECK_EVERY == 0:
                usage_now = du_bytes(managed_dirs(store))
        total_del += n
        total_freed += freed

    if total_del and not dry_run:
        remove_empty_dirs(store, log, dry_run)
    return total_del, total_freed


# ---------------------------------------------------------------------------
# docker-exec worker (firewatch etc.)
# ---------------------------------------------------------------------------
def run_worker(store, log, dry_run):
    cmd = store["exec_cmd"].split() + store["exec_args"]
    if not cmd:
        log(f"  store {store['name']}: TYPE=docker-exec but no EXEC_CMD")
        return 0, 0
    if dry_run and store["exec_dry_flag"]:
        cmd = cmd + [store["exec_dry_flag"]]
    log(f"  store {store['name']}: running {cmd}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"  store {store['name']}: worker failed to start: {exc}")
        return 0, 0
    tail = (result.stdout or "").strip().splitlines()[-5:]
    for line in tail:
        log(f"    {store['name']}| {line}")
    if result.returncode != 0:
        log(f"  store {store['name']}: worker exit {result.returncode} "
            f"(stderr: {(result.stderr or '').strip()[-200:]})")
    return 0, 0  # freed space is measured globally, not here


# ---------------------------------------------------------------------------
# phase 2: global-cap escalation
# ---------------------------------------------------------------------------
def phase2(stores, conf, log, dry_run, fs_free):
    """Free oldest files across stores (PRIORITY order) until free >= relief."""
    min_free = _float(conf.get("MIN_FREE_GB"), 0.0) * _GB
    used_pct = _float(conf.get("MAX_USED_PCT"), 0.0)
    relief = _float(conf.get("RELIEF_FREE_GB"), 0.0) * _GB
    fs_path = conf.get("FS_PATH", "")
    if relief <= 0 and min_free > 0:
        relief = min_free + 0.5 * _GB

    def free_gb():
        usage = fs_usage(fs_path or "/")
        return usage.free / _GB if usage else 0.0

    order = sorted(stores.values(), key=lambda s: s["priority"])
    log(f"escalation: free={fs_free / _GB:.2f}GiB -> target "
        f">= {relief / _GB:.2f}GiB; store order: "
        + ", ".join(f"{s['name']}(p{s['priority']})" for s in order))

    freed_total = 0
    del_total = 0
    for store in order:
        if free_gb() >= relief:
            break
        log(f"  escalate store {store['name']}")
        if store["type"] == "docker-exec":
            run_worker(store, log, dry_run)
            continue
        files, _total = collect_candidates(store, log, require_writable=True)
        if not files:
            log(f"    nothing deletable in {store['name']}")
            continue
        # delete oldest until the disk is relieved (recheck every N)
        n = 0
        while files and free_gb() < relief:
            f = files.pop(0)
            if dry_run:
                log(f"    dry: [escalate] {f['path']} "
                    f"({fmt_bytes(f['size'])})")
                n += 1
                del_total += 1
                continue
            try:
                os.unlink(f["path"])
            except OSError as exc:
                log(f"    {store['name']}: delete failed {f['path']}: {exc}")
                continue
            n += 1
            del_total += 1
            freed_total += f["size"]
            if n % _RECHECK_EVERY == 0:
                if free_gb() >= relief:
                    break
        if n and not dry_run:
            remove_empty_dirs(store, log, dry_run)
        log(f"    escalated {n} file(s) in {store['name']}")

    final_free = free_gb()
    log(f"escalation done: deleted={del_total} "
        f"freed={fmt_bytes(freed_total)} free now {final_free:.2f}GiB "
        f"(target {relief / _GB:.2f}GiB)")
    return del_total, freed_total, final_free


# ---------------------------------------------------------------------------
# permission pre-flight + --check table
# ---------------------------------------------------------------------------
def preflight(stores, log):
    """Verify each enabled store's managed dirs are reachable and, if they
    exist, writable. Returns (ok, warnings). Missing dirs are warnings (a
    service with no files yet must not fail the run); existing unwritable
    dirs are errors on a real run."""
    ok = True
    for store in stores.values():
        for d in managed_dirs(store):
            if not d:
                log(f"  preflight {store['name']}: no ROOT configured!")
                ok = False
                continue
            if not os.path.isdir(d):
                log(f"  preflight {store['name']}: missing dir {d} "
                    f"(ok - nothing stored yet)")
                continue
            if protected(d):
                log(f"  preflight {store['name']}: PROTECTED dir {d} "
                    f"- REFUSING")
                ok = False
                continue
            if not os.access(d, os.R_OK | os.X_OK):
                log(f"  preflight {store['name']}: dir {d} not "
                    f"readable/listable by euid {os.geteuid()} - ERROR")
                ok = False
            elif not is_writable_dir(d):
                log(f"  preflight {store['name']}: dir {d} not writable by "
                    f"euid {os.geteuid()} - ERROR (deleting inside needs "
                    f"write on the dir)")
                ok = False
            else:
                log(f"  preflight {store['name']}: {d} OK "
                    f"(r/w/x by euid {os.geteuid()})")
    return ok


def check_table(stores, fs_path, log):
    """--check: print a per-store table (no deletion)."""
    lines = []
    lines.append("Heartbeat cleanup --check "
                 f"(euid={os.geteuid()} fs={fs_path})")
    usage = fs_usage(fs_path or "/")
    if usage:
        pct = usage.used * 100.0 / usage.total if usage.total else 0.0
        lines.append(f"  disk: total={fmt_gb(usage.total / _GB)} "
                     f"used={fmt_gb(usage.used / _GB)} "
                     f"free={fmt_gb(usage.free / _GB)} "
                     f"({pct:.0f}% used)")
    header = ("{:<10} {:<6} {:<40} {:<8} {:<8} {:<8} {:<10} {}").format(
        "STORE", "TYPE", "DIR", "EXISTS", "READ", "WRITE", "USAGE",
        "CAPS(ageG/sizeG)")
    lines.append(header)
    lines.append("-" * len(header))
    for store in sorted(stores.values(), key=lambda s: s["name"]):
        dirs = managed_dirs(store)
        if not dirs:
            # docker-exec / no-ROOT store: show its exec command instead
            lines.append(
                ("{:<10} {:<6} {:<40} {:<8} {:<8} {:<8} {:<10} "
                 "{}").format(
                    store["name"][:10], store["type"][:6],
                    (store.get("exec_cmd") or "no ROOT")[:40],
                    "-", "-", "-", "-",
                    f"{store['max_age_days']:g}/{store['max_size_gb']:g}"))
            continue
        for d in dirs:
            exists = os.path.isdir(d)
            readable = os.access(d, os.R_OK | os.X_OK) if exists else False
            writable = is_writable_dir(d) if exists else False
            size = du_bytes([d]) if exists else 0
            lines.append(
                ("{:<10} {:<6} {:<40} {:<8} {:<8} {:<8} {:<10} "
                 "{}").format(
                    store["name"][:10], store["type"][:6], d[:40],
                    "yes" if exists else "no",
                    "yes" if readable else "no",
                    "yes" if writable else "no",
                    fmt_bytes(size),
                    f"{store['max_age_days']:g}/{store['max_size_gb']:g}"))
    for line in lines:
        log(line)


# ---------------------------------------------------------------------------
# telegram alert (optional, best-effort)
# ---------------------------------------------------------------------------
def maybe_alert(conf, text):
    if not _bool(conf.get("NOTIFY_TELEGRAM"), False):
        return
    try:
        import telegram_notify as tg  # local import (same dir)
        cfg = tg.load_conf(conf.get("TELEGRAM_CONF", ""))
        try:
            tg.ensure_creds(cfg)
        except RuntimeError as exc:
            print(f"[heartbeat] telegram not configured: {exc}",
                  file=sys.stderr)
            return
        tg.send_telegram(cfg, text)
    except Exception as exc:  # noqa: BLE001 - alerting must never break cron
        print(f"[heartbeat] telegram alert failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def default_deploy():
    return os.path.abspath(os.path.join(_HERE, ".."))


def main():
    ap = argparse.ArgumentParser(
        description="Unified disk heartbeat: per-service + global-cap cleanup.")
    ap.add_argument("--check", action="store_true",
                    help="print per-store permission/usage table, change nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted, change nothing")
    ap.add_argument("--deploy", default=None,
                    help="deploy root ({DEPLOY} token); default parent of script")
    ap.add_argument("--conf", default=None,
                    help="central heartbeat conf (default {DEPLOY}/config/heartbeat.conf)")
    ap.add_argument("--stores-dir", default=None,
                    help="dir of per-service store profiles")
    ap.add_argument("--log-file", default=None, help="override log file")
    ap.add_argument("--fs-path", default=None,
                    help="override the filesystem to watch for the global cap")
    args = ap.parse_args()

    deploy = args.deploy or os.environ.get("HEARTBEAT_DEPLOY") \
        or default_deploy()
    deploy = os.path.abspath(deploy)

    conf_path = args.conf or os.environ.get("HEARTBEAT_CONF") \
        or os.path.join(deploy, "config", "heartbeat.conf")
    conf = expand(read_conf(conf_path), deploy)
    if not conf:
        print(f"[heartbeat] ERROR: central conf not found/empty: "
              f"{conf_path}", file=sys.stderr)
        return 1

    stores_dir = args.stores_dir or os.environ.get("HEARTBEAT_STORES_DIR") \
        or conf.get("STORES_DIR") \
        or os.path.join(deploy, "config", "stores")
    log_file = args.log_file or conf.get("LOG_FILE") or ""
    fs_path = args.fs_path or conf.get("FS_PATH") or deploy

    log = Logger(log_file)
    require_root = _bool(conf.get("REQUIRE_ROOT"), True)
    dry_run = args.dry_run or _bool(conf.get("DRY_RUN"), False)

    stores = load_stores(stores_dir, deploy)
    if not stores:
        log(f"[heartbeat] WARNING: no enabled store profiles found in "
            f"{stores_dir}")
    log(f"[heartbeat] start deploy={deploy} stores_dir={stores_dir} "
        f"dry_run={dry_run} check={args.check}")

    # -- root requirement (real runs only) --------------------------------
    if not args.check and not dry_run and require_root and os.geteuid() != 0:
        log("[heartbeat] ERROR: real cleanup run requires root "
            f"(euid={os.geteuid()}); use --dry-run or --check to inspect, or "
            "run from the root crontab.")
        return 1

    # -- permission pre-flight --------------------------------------------
    preflight_ok = preflight(stores, log)

    # -- check mode: table + exit -----------------------------------------
    if args.check:
        check_table(stores, fs_path, log)
        log("[heartbeat] check done - nothing changed")
        return 0 if preflight_ok else 1

    # -- measure the global filesystem ------------------------------------
    usage = fs_usage(fs_path or "/")
    if usage is None:
        log(f"[heartbeat] ERROR: cannot stat fs {fs_path}")
        return 1
    total_b, used_b, free_b = usage
    used_pct = used_b * 100.0 / total_b if total_b else 0.0
    min_free = _float(conf.get("MIN_FREE_GB"), 0.0) * _GB
    used_cap_pct = _float(conf.get("MAX_USED_PCT"), 0.0)
    log(f"[heartbeat] fs={fs_path} total={fmt_bytes(total_b)} "
        f"used={fmt_bytes(used_b)} free={fmt_bytes(free_b)} "
        f"({used_pct:.1f}% used) min_free={fmt_bytes(min_free)}")

    # -- Phase 1: per-store compliance ------------------------------------
    total_del = 0
    total_freed = 0
    for service, store in sorted(stores.items(), key=lambda kv: kv[1]["name"]):
        n, freed = phase1_store(store, log, dry_run)
        total_del += n
        total_freed += freed

    # -- Phase 2: global-cap escalation -----------------------------------
    triggered = (min_free > 0 and free_b < min_free) or \
                (used_cap_pct > 0 and used_pct >= used_cap_pct)
    if triggered:
        log(f"[heartbeat] GLOBAL CAP hit "
            f"(free {free_b / _GB:.2f}GiB < {min_free / _GB:.2f}GiB "
            f"or {used_pct:.1f}% >= {used_cap_pct:g}%) - escalating")
        esc_del, esc_freed, final_free = phase2(stores, conf, log, dry_run,
                                                free_b)
        total_del += esc_del
        total_freed += esc_freed
        summary = (f"<b>🛑 Disk heartbeat escalation</b>\n"
                   f"Disk on {fs_path} hit the global cap "
                   f"(free {free_b / _GB:.1f}GiB / {used_pct:.0f}% used).\n"
                   f"Deleted {esc_del} file(s) across services, freed "
                   f"{fmt_bytes(esc_freed)} -> {final_free:.1f}GiB free "
                   f"(target {_float(conf.get('RELIEF_FREE_GB'), 0):g}GiB).")
        maybe_alert(conf, summary)
    elif not preflight_ok:
        maybe_alert(conf,
                    "<b>⚠️ Disk heartbeat pre-flight failure</b>\n"
                    "Some store dirs are missing/unwritable - see the log.")

    log(f"[heartbeat] done: deleted={total_del} "
        f"freed={fmt_bytes(total_freed)} "
        f"preflight_ok={preflight_ok}")
    return 0 if preflight_ok else 2


if __name__ == "__main__":
    sys.exit(main())
