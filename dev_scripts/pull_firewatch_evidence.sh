#!/usr/bin/env bash
# pull_firewatch_evidence.sh - download EVERY image in the firewatch evidence
# store, plus a consistent snapshot of its DB and the frame/detection metadata.
#
# WHAT "firewatch images" MEANS
# -----------------------------
# firewatch persists evidence as plain JPEGs at <STORE_DIR>/<cam>/<base>.jpg
# (the ORIGINAL the model scored) and <base>_annotated.jpg (box overlay), with
# one `frames` row per stored frame in <STORE_DIR>/firewatch.db.  <STORE_DIR>
# is /media/firewatch in the container = the deploy's ./media on the host, which
# is ALSO Frigate's media mount - so the store shares a tree with Frigate's
# clips/recordings.  This script therefore selects:
#   * every file the DB references (jetson/normal names), AND
#   * every UNREFERENCED file that still matches firewatch's own naming pattern
#     `YYYYMMDD_HHMMSS_mmm_camNN_conf<score>[_annotated].jpg` - these orphans are
#     exactly what the 2026-09-12 readonly-WAL incident left behind (JPEG written,
#     row lost), so they are evidence too.
# Frigate's clips/, events/, exports/, portal/, scenewatch/, watchdog/ files match
# neither test and are NOT downloaded.
#
# WHY THE TWO-HOP DANCE (container -> host /tmp -> local)
# -----------------------------------------------------
# The live evidence DB must NEVER be opened from the host: a host-side open (even
# a read-only mode=ro one) creates -wal/-shm OWNED BY THE HOST USER and silently
# wedges every later firewatch write (2 days of evidence were lost this way; see
# .roo/rules/firewatch-store-ownership.md).  So ALL DB access happens INSIDE the
# firewatch container, which is the store's legitimate (uid 1000) writer:
#   step 1  container: read the DB read-only, hardlink/copy every evidence JPEG
#           into a container stage dir, write manifest.csv (with per-file md5),
#           INVENTORY.txt, frames.csv, detections.csv and a consistent DB copy
#           (SQLite `VACUUM INTO`, which folds in any pending -wal);
#   step 2  `docker cp` the stage to the host's /tmp - the AI user is NOT the
#           owner of the deploy dir, so nothing is ever written there;
#   step 3  rsync the stage down to ./firewatch-evidence/ (resumable);
#   step 4  verify every local file's size + md5 against manifest.csv;
#   step 5  remove both staging dirs (unless --keep-stage).
#
# Usage:
#   DEPLOY_SSH_USER=ai DEPLOY_SSH_PASS='...' dev_scripts/pull_firewatch_evidence.sh
#   ... --dry-run       stage + show what rsync WOULD fetch, then clean up
#   ... --keep-stage    leave the container/host staging dirs in place
#
# Env (defaults in []):
#   LOCAL_DIR          local mirror            [./firewatch-evidence]
#   FW_CONTAINER       firewatch container     [firewatch]
#   FW_STORE_PARENT    parent of the store dir [/media]        (container path)
#   FW_STORE_NAME      store dir name          [firewatch]     (container path)
#   FW_CTR_STAGE       container staging dir   [/tmp/fw_evidence_export]
#   FW_HOST_STAGE      host staging dir        [/tmp/fw_evidence_export]
#   SSH_HOST / SSH_USER / SSH_PASS   (or DEPLOY_SSH_HOST / DEPLOY_SSH_USER / DEPLOY_SSH_PASS)
set -uo pipefail

LOCAL_DIR="${LOCAL_DIR:-./firewatch-evidence}"
FW_CONTAINER="${FW_CONTAINER:-firewatch}"
FW_STORE_PARENT="${FW_STORE_PARENT:-/media}"
FW_STORE_NAME="${FW_STORE_NAME:-firewatch}"
FW_CTR_STAGE="${FW_CTR_STAGE:-/tmp/fw_evidence_export}"
FW_HOST_STAGE="${FW_HOST_STAGE:-/tmp/fw_evidence_export}"
SSH_HOST="${DEPLOY_SSH_HOST:-${SSH_HOST:-ssh.mazr3a.garden}}"
SSH_USER="${DEPLOY_SSH_USER:-${SSH_USER:-}}"
SSH_PASS="${DEPLOY_SSH_PASS:-${SSH_PASSWORD:-}}"

DRY_RUN=0
KEEP_STAGE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY_RUN=1 ;;
    --keep-stage) KEEP_STAGE=1 ;;
    -h|--help)    sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ -z "$SSH_USER" ] || [ -z "$SSH_PASS" ]; then
  echo "ERROR: set DEPLOY_SSH_USER and DEPLOY_SSH_PASS (or SSH_USER/SSH_PASSWORD)" >&2
  exit 1
fi
command -v rsync >/dev/null || { echo "ERROR: rsync not installed locally" >&2; exit 1; }

# --- SSH_ASKPASS setup (verbatim password file, no shell re-interpretation) ---
PASSFILE="$(mktemp)"; printf '%s\n' "$SSH_PASS" > "$PASSFILE"; chmod 600 "$PASSFILE"
ASKPASS="$(mktemp)"
printf '#!/usr/bin/env bash\ncat "%s"\n' "$PASSFILE" > "$ASKPASS"; chmod 700 "$ASKPASS"
trap 'rm -f "$ASKPASS" "$PASSFILE"' EXIT
export SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force DISPLAY=:0

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=25
  -o ServerAliveInterval=15 -o ServerAliveCountMax=6)
SSH_CMD="/usr/bin/ssh ${SSH_OPTS[*]}"

run_ssh() { # "<remote-command>": retry up to 5x (mirrors deploy_all.sh run_ssh)
  local cmd="$1" n=1
  until setsid /usr/bin/ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$cmd"; do
    echo "   (ssh retry ${n}/5)" >&2
    if [ "$n" -ge 5 ]; then echo "ERROR: ssh failed after 5 attempts" >&2; return 1; fi
    n=$((n + 1)); sleep 10
  done
}

# ---------------------------------------------------------------------------
# 1) container-side staging + inventory (the ONLY place the DB is opened)
# ---------------------------------------------------------------------------
FW_PY="$(base64 -w0 <<'PY'
import csv, hashlib, os, re, shutil, sqlite3, sys

STORE = os.path.join(os.sep, os.environ.get("FW_STORE_PARENT", "/media").strip(os.sep),
                     os.environ.get("FW_STORE_NAME", "firewatch"))
STAGE = os.environ.get("FW_CTR_STAGE", "/tmp/fw_evidence_export")
DB = os.path.join(STORE, "firewatch.db")
NAME_RE = re.compile(r"^\d{8}_\d{6}_\d{3}_cam[0-9]+_conf[0-9.]+(_annotated)?\.jpe?g$", re.I)

if not os.path.isfile(DB):
    sys.exit("ERROR: no evidence DB at %s" % DB)
if os.path.isdir(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)

# READ-ONLY inside the container: any -wal/-shm this creates is owned by the
# container user (uid 1000) = the store's legitimate writer, so it is harmless.
con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
con.row_factory = sqlite3.Row
frames = [dict(r) for r in con.execute("SELECT * FROM frames ORDER BY id")]
try:
    dets = [dict(r) for r in con.execute("SELECT * FROM detections ORDER BY id")]
    det_cols = list(dets[0].keys()) if dets else \
        [r[1] for r in con.execute("PRAGMA table_info(detections)")]
except sqlite3.Error:
    dets, det_cols = [], []
frm_cols = [r[1] for r in con.execute("PRAGMA table_info(frames)")]

ref = {}
for fr in frames:
    for kind, key in (("raw", "jpg_path"), ("annotated", "annotated_path")):
        p = fr.get(key)
        if p:
            ref[os.path.normpath(p)] = (fr["id"], fr.get("camera") or "", kind)

def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
for entry in sorted(os.listdir(STORE)):
    cam_dir = os.path.join(STORE, entry)
    if not os.path.isdir(cam_dir):
        continue
    for name in sorted(os.listdir(cam_dir)):
        src = os.path.join(cam_dir, name)
        if not os.path.isfile(src):
            continue
        hit = ref.get(os.path.normpath(src))
        if hit is None and not NAME_RE.match(name):
            continue            # Frigate / another service's file - not evidence
        out_dir = os.path.join(STAGE, entry)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, name)
        try:
            os.link(src, dst)           # cheap when same fs; EXDEV -> copy
        except OSError:
            shutil.copy2(src, dst)
        rows.append({
            "file": os.path.join(entry, name),
            "camera": (hit[1] if hit else entry),
            "kind": (hit[2] if hit else
                     ("annotated" if name.endswith("_annotated.jpg") else "raw")),
            "in_db": "1" if hit else "0",
            "frame_id": (hit[0] if hit else ""),
            "bytes": os.path.getsize(src),
            "md5": md5_of(dst) if not os.path.samefile(src, dst) else md5_of(src),
        })

# --- metadata tables (no host-side DB open is ever needed) ------------------
with open(os.path.join(STAGE, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "camera", "kind", "in_db", "frame_id",
                                       "bytes", "md5"])
    w.writeheader()
    w.writerows(rows)
for name, data, cols in (("frames.csv", frames, frm_cols),
                         ("detections.csv", dets, det_cols)):
    if not cols:
        continue
    with open(os.path.join(STAGE, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for d in data:
            w.writerow({k: d.get(k) for k in cols})

# --- consistent DB snapshot (folds in the pending -wal) --------------------
snap = os.path.join(STAGE, "firewatch.db")
con2 = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
con2.execute("VACUUM INTO ?", (snap,))
con2.close()

# --- INVENTORY.txt ---------------------------------------------------------
total_b = sum(r["bytes"] for r in rows)
in_db = [r for r in rows if r["in_db"] == "1"]
orphans = [r for r in rows if r["in_db"] == "0"]
per_cam = {}
for r in rows:
    st = per_cam.setdefault(r["camera"], [0, 0, 0])
    st[0] += 1
    st[1] += r["bytes"]
    st[2] += 1 if r["in_db"] == "0" else 0
lines = []
lines.append("firewatch evidence export")
lines.append("store (container): %s" % STORE)
lines.append("db (container)   : %s" % DB)
lines.append("")
lines.append("frames rows      : %d" % len(frames))
lines.append("detections rows  : %d" % len(dets))
lines.append("images exported  : %d  (%.1f MiB)" % (len(rows), total_b / 1048576.0))
lines.append("  ... in DB      : %d  (%.1f MiB)"
             % (len(in_db), sum(r["bytes"] for r in in_db) / 1048576.0))
lines.append("  ... ORPHANS    : %d  (%.1f MiB)  <- on disk, no DB row"
             % (len(orphans), sum(r["bytes"] for r in orphans) / 1048576.0))
lines.append("")
lines.append("%-8s %8s %10s %8s" % ("camera", "images", "MiB", "orphans"))
for cam in sorted(per_cam):
    n, b, o = per_cam[cam]
    lines.append("%-8s %8d %10.1f %8d" % (cam, n, b / 1048576.0, o))
with open(os.path.join(STAGE, "INVENTORY.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

print("\n".join(lines))
print("")
print("manifest.csv sha256: %s" % hashlib.sha256(
    open(os.path.join(STAGE, "manifest.csv"), "rb").read()).hexdigest())
print("STAGED %d images, %d frames, %d detections in %s"
      % (len(rows), len(frames), len(dets), STAGE))
PY
)"

echo "=== 1/5 staging evidence INSIDE the ${FW_CONTAINER} container (DB read read-only) ==="
if ! run_ssh "base64 -d <<< '${FW_PY}' | docker exec -i \
  -e FW_STORE_PARENT='${FW_STORE_PARENT}' -e FW_STORE_NAME='${FW_STORE_NAME}' \
  -e FW_CTR_STAGE='${FW_CTR_STAGE}' ${FW_CONTAINER} python3 -"; then
  echo "ERROR: container-side staging failed" >&2
  exit 1
fi

echo
echo "=== 2/5 copying the stage to the host's ${FW_HOST_STAGE} ==="
run_ssh "rm -rf '${FW_HOST_STAGE}' && mkdir -p '${FW_HOST_STAGE}' && \
  docker cp '${FW_CONTAINER}:${FW_CTR_STAGE}/.' '${FW_HOST_STAGE}/' && \
  find '${FW_HOST_STAGE}' -type f | wc -l" || exit 1

mkdir -p "$LOCAL_DIR"
echo
echo "=== 3/5 rsync -> ${LOCAL_DIR}/ ==="
RSYNC_FLAGS=(-rlt --no-perms --no-owner --no-group --partial --human-readable)
[ "$DRY_RUN" = "1" ] && RSYNC_FLAGS+=(--dry-run -i)
/usr/bin/rsync -e "$SSH_CMD" "${RSYNC_FLAGS[@]}" \
  "${SSH_USER}@${SSH_HOST}:${FW_HOST_STAGE}/" "$LOCAL_DIR/"
rc=$?
echo "rsync rc=$rc"
if [ "$rc" -ne 0 ]; then echo "ERROR: rsync failed - staging dirs kept" >&2; exit $rc; fi

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "--dry-run: staging dirs removed, nothing kept locally."
fi

echo
echo "=== 4/5 verifying size + md5 against manifest.csv ==="
python3 - "$LOCAL_DIR" <<'PY'
import csv, hashlib, os, sys
root = sys.argv[1]
mpath = os.path.join(root, "manifest.csv")
if not os.path.isfile(mpath):
    sys.exit("FAILED: manifest.csv missing in %s" % root)
bad = miss = 0
total = 0
n = 0
with open(mpath, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        n += 1
        p = os.path.join(root, row["file"])
        if not os.path.isfile(p):
            miss += 1
            print("  MISSING %s" % row["file"])
            continue
        sz = os.path.getsize(p)
        total += sz
        h = hashlib.md5()
        with open(p, "rb") as f2:
            for chunk in iter(lambda: f2.read(1 << 20), b""):
                h.update(chunk)
        if sz != int(row["bytes"]) or h.hexdigest() != row["md5"]:
            bad += 1
            print("  CORRUPT %s (size %d/%s md5 %s/%s)"
                  % (row["file"], sz, row["bytes"], h.hexdigest(), row["md5"]))
print("checked %d files, %.1f MiB - missing %d, corrupt %d"
      % (n, total / 1048576.0, miss, bad))
sys.exit(1 if (miss or bad) else 0)
PY
verify_rc=$?

if [ "$KEEP_STAGE" = "1" ]; then
  echo
  echo "=== 5/5 --keep-stage: staging dirs LEFT in place ==="
  echo "    container: ${FW_CTR_STAGE}   host: ${FW_HOST_STAGE}"
else
  echo
  echo "=== 5/5 cleaning staging dirs ==="
  run_ssh "docker exec ${FW_CONTAINER} rm -rf '${FW_CTR_STAGE}'; rm -rf '${FW_HOST_STAGE}'" \
    || echo "WARN: could not clean staging dirs" >&2
fi

echo
if [ "$verify_rc" -ne 0 ]; then
  echo "RESULT: FAILED verification - re-run this script (rsync is resumable)."
  exit "$verify_rc"
fi
echo "RESULT: OK - evidence at ${LOCAL_DIR}/ (images + manifest.csv + INVENTORY.txt"
echo "        + frames.csv + detections.csv + a consistent firewatch.db snapshot)."
