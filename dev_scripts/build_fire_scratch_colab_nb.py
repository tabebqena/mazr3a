#!/usr/bin/env python3
"""build_fire_scratch_colab_nb.py - generate notebooks/fire-scratch-train-colab.ipynb.

A ready-to-import Google Colab notebook for the "scratch" fire-model campaign: it trains from a
COCO-pretrained backbone (``yolo11s.pt``) - NOT from the production fire checkpoint
(``models/fire/best.pt``, i.e. v4/v5/v6) - on external datasets it downloads itself (nothing large
is uploaded), then deduplicates them honestly and trains in a way that survives losing the editor.
A later campaign run can continue from the previous run's winner (e.g. v2 continues
``v1/scratch-v1.pt``).

THE HONEST-VALIDATION CONTRACT (the whole point of this notebook)
-----------------------------------------------------------------
  1. download external training sets from their own sources (HF corpus / zip URLs);
  2. dedup TRAIN against itself FIRST (so the model generalises, it does not memorise);
  3. dedup TEST against itself FIRST (one image must not win many points);
  4. dedup TEST images against TRAIN only AFTER each set is internally clean
     (a memorised test image would flatter the score);
  5. on EVERY run, dedup against the PREVIOUS runs' sets via a LIGHTWEIGHT fingerprint
     index on Drive (md5 + 64-bit dHash, ~100 bytes/image) - old datasets are never
     re-downloaded, only their fingerprints are compared;
  6. our CCTV images go to the held-out test split ONLY (role=test) and are never trained.

SURVIVES A RUNTIME RESET / DISCONNECT
-------------------------------------
  * run artifacts land on Drive: <DRIVE>/runs/<name>/ (train.log, results.csv, args.yaml,
    weights/{last,best}.pt), mirrored every epoch;
  * training is launched DETACHED (subprocess.Popen(start_new_session=True) -> the bundled
    dev_scripts/colab_train_scratch.py), so closing VS Code / the tab cannot stop it;
  * a RESUME cell restarts an interrupted run from last.pt.

INCREMENTAL IMPROVEMENT / REGRESSION DECISION
---------------------------------------------
  The COLLECT cell scores the run on the held-out test split, appends the run to
  <DRIVE>/registry.json, and compares its mAP@50 to the previous best. If it regressed, the
  previous best is kept active and the decision (retrain previous vs continue) is printed for
  a human call - nothing is silently overwritten.

Usage:
    python dev_scripts/build_fire_scratch_colab_nb.py
        # writes notebooks/fire-scratch-train-colab.ipynb
    python dev_scripts/build_fire_scratch_colab_nb.py --out <dir>/fire-scratch-train-colab.ipynb
"""
import argparse
import json
import os

DEFAULT_OUT = "notebooks/fire-scratch-train-colab.ipynb"

# Bump on every notebook change (it is stamped into the notebook title + metadata).
NOTEBOOK_VERSION = "1.4.0"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


C_TITLE = md([
    "# Fire model — scratch campaign (COCO backbone, honest validation)\n",
    "\n",
    "Trains a fire detector **from a COCO-pretrained backbone** (`yolo11s.pt`) — **not** from the\n",
    "production fire checkpoint (`models/fire/best.pt`, i.e. v4/v5/v6). Later campaign runs can\n",
    "continue from the previous run's winner (e.g. `v2` continues `v1/scratch-v1.pt`). Datasets are\n",
    "downloaded by the notebook itself, deduplicated honestly, and artifacts live on Drive.\n",
    "\n",
    "- **Dataset build:** download → merge → dedup → clean `train/val/test`. Dedup runs each set\n",
    "  against ITSELF first, then cross-set (see the honest-validation contract below).\n",
    "- **Every run dedups against the PREVIOUS runs** via a lightweight fingerprint index on\n",
    "  Drive (`md5` + 64-bit dHash, ~100 bytes/image). Old datasets are never re-downloaded.\n",
    "- **CCTV images are test-only** (role=test) and never enter training.\n",
    "- **Training is detached** (`subprocess.Popen(start_new_session=True)`): closing VS Code /\n",
    "  the tab cannot stop it, and the per-epoch checkpoint is mirrored to Drive.\n",
    "- **Incremental improvement:** the COLLECT cell scores the run on the held-out test split,\n",
    "  compares it to the previous best in `<DRIVE>/registry.json`, and promotes or rejects.\n",
    "\n",
    "Plan: `plans/fire-model-scratch-colab.md`.\n",
])

C_SETUP = code([
    "# Cell 1 - deps + GPU\n",
    "!nvidia-smi\n",
    "!pip -q install --upgrade ultralytics pyarrow huggingface_hub gdown kagglehub\n",
    "import torch, ultralytics, pyarrow, huggingface_hub\n",
    "print('ultralytics', ultralytics.__version__, '| pyarrow', pyarrow.__version__)\n",
    "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())\n",
    "assert torch.cuda.is_available(), 'pick a GPU runtime (Runtime -> Change runtime type)'\n",
])

C_CONFIG = code([
    "# Cell 2 - EDIT THIS (paths, classes, sources, hyperparameters)\n",
    "\n",
    "DRIVE_DIR  = '/content/drive/MyDrive/mazr3a-fire-scratch'   # <-- your Drive folder\n",
    "VERSION    = 'v2'                     # THIS run's campaign folder (v1 = scratch-v1, v2 = this run)\n",
    "PREV_VER   = 'v1'                     # folder of the run we CONTINUE from (base model + fingerprints)\n",
    "DRIVE_ZIP  = DRIVE_DIR + '/colab_upload.zip'               # bundle from pack_fire_scratch_colab.sh (shared)\n",
    "RUNS       = DRIVE_DIR + '/' + VERSION + '/runs'           # THIS run's artifacts (mirrored every epoch)\n",
    "FPS_TRAIN  = DRIVE_DIR + '/' + VERSION + '/fingerprints/train'  # THIS run's train index (written by COLLECT)\n",
    "FPS_TEST   = DRIVE_DIR + '/' + VERSION + '/fingerprints/test'   # THIS run's test/val index (written by COLLECT)\n",
    "PREV_TRAIN = DRIVE_DIR + '/' + PREV_VER + '/fingerprints/train' # dedup vs previous TRAIN images\n",
    "PREV_TEST  = DRIVE_DIR + '/' + PREV_VER + '/fingerprints/test'  # dedup vs previous held-out TEST/val images\n",
    "REGISTRY   = DRIVE_DIR + '/' + VERSION + '/registry.json'  # run history + promote/regress decisions\n",
    "PREV_REG   = DRIVE_DIR + '/' + PREV_VER + '/registry.json' # previous run's history (seed for comparison)\n",
    "LOGS       = DRIVE_DIR + '/' + VERSION + '/logs'\n",
    "\n",
    "LOCAL_RUNS = '/content/runs'        # fast local project dir (Drive FUSE is slow)\n",
    "RUNS_MODE  = 'local_mirror'         # 'local_mirror' (default) | 'drive'\n",
    "LOG_LOCAL  = '/content/train.log'   # detached trainer stdout (mirrored to Drive)\n",
    "NAME       = 'scratch-v1-dfire'     # run name -> RUNS/<name>/weights/...\n",
    "PIDFILE    = '/content/train.pid'\n",
    "\n",
    "# --- VM paths (idempotent) ---\n",
    "UPLOAD = '/content/upload'          # unpacked colab_upload.zip\n",
    "CACHE  = '/content/src_cache'       # downloaded sources (temporary)\n",
    "RAW    = '/content/raw_yolo'        # merged pool (before dedup)\n",
    "CLEAN  = '/content/clean_yolo'      # clean pool (after dedup) - what we train on\n",
    "\n",
    "# --- class contract ---\n",
    "CLASSES   = ['fire']                    # fire-only first (your lean).\n",
    "                                        #   ['fire','smoke']         -> fire + smoke\n",
    "                                        #   ['fire','other','smoke'] -> production contract\n",
    "CLASS_MAP = {                           # source class name -> target class (None drops the box)\n",
    "    'fire': 'fire', 'Fire': 'fire',\n",
    "    'flame_visible': 'fire',\n",
    "    'smoke': None, 'smoke_visible': None,   # fire-only: smoke boxes dropped -> those imgs train as background\n",
    "    'other': None, 'default': None,\n",
    "}\n",
    "# NOTE on smoke: dropping smoke boxes only affects THIS run's labels. The source data is\n",
    "# untouched, so a later run with CLASSES=['fire','smoke'] and 'smoke': 'smoke' rebuilds the\n",
    "# smoke boxes. The persistent fingerprint index also only blocks images previously trained\n",
    "# as a POSITIVE class, so smoke images used as background here stay re-usable as smoke later.\n",
    "\n",
    "# --- base model (continued FINE-TUNE of run 1's winner; NOT from scratch) ---\n",
    "BASE_MODEL = DRIVE_DIR + '/' + PREV_VER + '/scratch-v1.pt'   # run 1's fire-only winner (nc=1)\n",
    "EXPECTED_BASE_MD5 = '6e3325cca6d07dc58028218b58d3e746'  # md5 of the LOCAL v1 copy - launch aborts on mismatch\n",
    "\n",
    "# --- training sources: D-Fire only (pulled by Colab from Kaggle, not bundled) ---\n",
    "SOURCES = [\n",
    "    # D-Fire Kaggle 'ready to use' (CC0), the official README's mirror of the OneDrive files.\n",
    "    # role='train' maps D-Fire train->train, val->val, test->test (the test split stays held out).\n",
    "    {'id': 'dfire', 'type': 'kaggle_yolo', 'repo': 'sayedgamal99/smoke-fire-detection-yolo',\n",
    "     'role': 'train', 'index_map': {0: None, 1: 'fire'}},   # 0=smoke dropped, 1=fire -> fire(0)\n",
    "    # Domain negatives intentionally NOT re-added (keep the D-Fire-only signal). Re-add later:\n",
    "    # {'id': 'negatives', 'type': 'yolo_dir', 'path': UPLOAD + '/negatives', 'role': 'negatives'},\n",
    "]\n",
    "\n",
    "# --- dedup ---\n",
    "DEDUP_HAMMING     = 8      # dHash Hamming distance; 6 conservative, 10 aggressive\n",
    "DEDUP_TRAIN_SCOPE = 'group'  # 'group' = within a clip (video frames); 'global' = whole train set\n",
    "\n",
    "# --- fire vs not-fire balance (decides the positive/background mix of TRAIN) ---\n",
    "MAX_BG_SHARE = 0.60   # cap the background (non-fire) fraction of TRAIN at 60%\n",
    "                      #   raise to 0.70-0.80 if false positives dominate in the pilot;\n",
    "                      #   lower to 0.50 if fire recall lags. Excess background is moved to\n",
    "                      #   <clean>_bg_extra (reversible) - never deleted.\n",
    "\n",
    "# --- training (continued FINE-TUNE of scratch-v1, not from-scratch) ---\n",
    "EPOCHS      = 50\n",
    "BATCH       = 16    # 32 on an L4, 64+ on an A100; 8 if a T4 OOMs\n",
    "IMGSZ       = 640   # matches production input\n",
    "FREEZE      = 10    # freeze the backbone (fine-tune convention; scratch runs use 0)\n",
    "LR0         = 0.001 # fine-tune LR (scratch runs use 0.01)\n",
    "PATIENCE    = 15    # early stop on the held-out val split\n",
    "SAVE_PERIOD = 1     # checkpoint every epoch -> survives a recycle\n",
    "\n",
    "# --- per-cell prerequisite guard (every cell verifies the cells it depends on ran) ---\n",
    "def need(ok, what):\n",
    "    if not ok:\n",
    "        raise SystemExit('PREREQUISITE MISSING - ' + what)\n",
    "\n",
    "# --- training status (success/failure, not just 'finished?') ---\n",
    "def training_status():\n",
    "    \"\"\"Return (state, done_epochs, log_tail) for the DETACHED trainer.\n",
    "    state in: none | running | completed | early-stopped | failed | unknown\"\"\"\n",
    "    import os\n",
    "    rows = []\n",
    "    for p in (os.path.join(LOCAL_RUNS, NAME, 'results.csv'),\n",
    "              os.path.join(RUNS, NAME, 'results.csv')):\n",
    "        if os.path.exists(p):\n",
    "            rows = [r for r in open(p, errors='replace').read().splitlines() if r.strip()]\n",
    "            break\n",
    "    n = max(0, len(rows) - 1)\n",
    "    pid = int(open(PIDFILE).read().strip() or 0) if os.path.exists(PIDFILE) else 0\n",
    "    running = False\n",
    "    if pid and os.path.exists('/proc/%d' % pid):\n",
    "        try:\n",
    "            running = open('/proc/%d/stat' % pid).read().split()[2] != 'Z'\n",
    "        except (FileNotFoundError, IndexError):\n",
    "            running = False\n",
    "    log = LOG_LOCAL\n",
    "    if not os.path.exists(log):\n",
    "        log = os.path.join(RUNS, NAME, 'train.log')\n",
    "    tail = ''\n",
    "    if os.path.exists(log):\n",
    "        tail = '\\n'.join(open(log, errors='replace').read().splitlines()[-25:])\n",
    "    if running:\n",
    "        return 'running', n, tail\n",
    "    if n >= EPOCHS:\n",
    "        return 'completed', n, tail\n",
    "    low = tail.lower()\n",
    "    if ('traceback' in low or 'cuda out of memory' in low\n",
    "            or 'killed' in low or 'error:' in low):\n",
    "        return 'failed', n, tail\n",
    "    if ('earlystop' in low or 'no improvement observed' in low\n",
    "            or 'stopping training' in low):\n",
    "        return 'early-stopped', n, tail\n",
    "    if not rows and pid == 0:\n",
    "        return 'none', n, tail\n",
    "    return 'unknown', n, tail\n",
])

C_LOGGING = code([
    "# Cell 3 - mount Drive + debug logging (every stage's stdout also lands in <DRIVE>/logs/)\n",
    "import contextlib, os, sys, time\n",
    "from google.colab import drive\n",
    "\n",
    "drive.mount('/content/drive')\n",
    "for d in (DRIVE_DIR, RUNS, FPS_TRAIN, FPS_TEST, LOGS):\n",
    "    os.makedirs(d, exist_ok=True)\n",
    "\n",
    "class _Tee:\n",
    "    \"\"\"Duplicate everything written to stdout into a file (line-buffered) on Drive.\"\"\"\n",
    "    def __init__(self, path):\n",
    "        self.path = path\n",
    "        self.f = open(path, 'a', buffering=1)\n",
    "        self.out = getattr(sys.stdout, 'out', sys.stdout)\n",
    "    def write(self, s):\n",
    "        self.f.write(s)\n",
    "        try:\n",
    "            self.out.write(s)\n",
    "        except Exception:\n",
    "            pass\n",
    "    def flush(self):\n",
    "        try:\n",
    "            self.f.flush()\n",
    "            self.out.flush()\n",
    "        except Exception:\n",
    "            pass\n",
    "    def close(self):\n",
    "        try:\n",
    "            self.f.close()\n",
    "        except Exception:\n",
    "            pass\n",
    "\n",
    "@contextlib.contextmanager\n",
    "def drive_log(stage):\n",
    "    \"\"\"with drive_log('07_dedup'): <work>  -> full stdout also lands in <DRIVE>/logs/.\"\"\"\n",
    "    path = os.path.join(LOGS, stage + '.log')\n",
    "    tee = _Tee(path)\n",
    "    tee.write('\\n' + '=' * 72 + '\\n[%s] started %s\\n' % (stage, time.strftime('%Y-%m-%d %H:%M:%S'))\n",
    "              + '=' * 72 + '\\n')\n",
    "    try:\n",
    "        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):\n",
    "            yield path\n",
    "    finally:\n",
    "        tee.write('[%s] finished %s\\n' % (stage, time.strftime('%H:%M:%S')))\n",
    "        tee.close()\n",
    "    print('log ->', path)\n",
    "\n",
    "print('logs ->', LOGS)\n",
])

C_BUNDLE = code([
    "# Cell 4 - unpack the (small) upload bundle from Drive (self-healing)\n",
    "import glob, os, shutil, zipfile\n",
    "need(os.path.exists('/content/drive/MyDrive'), 'run Cell 3 (mount Drive) first')\n",
    "\n",
    "REQUIRED = ['scripts/prep_fire_scratch_dataset.py', 'scripts/prep_fireviewer_dataset.py',\n",
    "            'scripts/dedup_fire_scratch.py', 'scripts/colab_train_scratch.py']\n",
    "os.makedirs(os.path.join(RUNS, NAME), exist_ok=True)   # the log needs a home on Drive\n",
    "\n",
    "missing = [r for r in REQUIRED if not os.path.exists(os.path.join(UPLOAD, r))]\n",
    "if missing:\n",
    "    assert os.path.exists(DRIVE_ZIP), ('missing ' + DRIVE_ZIP +\n",
    "        ' - build it with dev_scripts/pack_fire_scratch_colab.sh and upload it to ' + DRIVE_DIR)\n",
    "    print('(re)extracting bundle; missing so far:', ', '.join(missing))\n",
    "    if os.path.isdir(UPLOAD):\n",
    "        shutil.rmtree(UPLOAD)\n",
    "    os.makedirs(UPLOAD, exist_ok=True)\n",
    "    with zipfile.ZipFile(DRIVE_ZIP) as z:\n",
    "        z.extractall(UPLOAD)\n",
    "        if 'BUNDLE.txt' in z.namelist():\n",
    "            print('--- bundle provenance ---')\n",
    "            print(z.read('BUNDLE.txt').decode().strip())\n",
    "            print('-------------------------')\n",
    "\n",
    "missing = [r for r in REQUIRED if not os.path.exists(os.path.join(UPLOAD, r))]\n",
    "assert not missing, ('bundle is missing ' + ', '.join(missing) +\n",
    "                     ' - rebuild with dev_scripts/pack_fire_scratch_colab.sh, re-upload to '\n",
    "                     + DRIVE_DIR + ', then re-run this cell')\n",
    "print('bundle:', sorted(os.listdir(UPLOAD)))\n",
    "print('domain_test:', len(glob.glob(UPLOAD + '/domain_test/*')), '| negatives:', len(glob.glob(UPLOAD + '/negatives/*')))\n",
])

C_SOURCES = code([
    "# Cell 5 - build the RAW merged pool: download external sources -> remap classes -> merge\n",
    "# IDEMPOTENT: re-running after a reset is a no-op if RAW/data.yaml already exists.\n",
    "import json, os, subprocess, sys\n",
    "need(os.path.exists(UPLOAD + '/scripts/prep_fire_scratch_dataset.py'), 'run Cell 4 (unpack bundle) first')\n",
    "\n",
    "with drive_log('05_sources'):\n",
    "    cfg = {'classes': CLASSES, 'class_map': CLASS_MAP, 'sources': SOURCES}\n",
    "    cfg_path = '/content/sources.json'\n",
    "    json.dump(cfg, open(cfg_path, 'w', encoding='utf-8'), indent=2)\n",
    "    prep = UPLOAD + '/scripts/prep_fire_scratch_dataset.py'\n",
    "    assert os.path.exists(prep), 'bundle missing prep script - see Cell 4'\n",
    "    subprocess.run([sys.executable, prep, '--config', cfg_path, '--out', RAW,\n",
    "                    '--cache', CACHE, '--scripts', UPLOAD + '/scripts'], check=True)\n",
    "    print(open(RAW + '/prep_report.txt', encoding='utf-8').read())\n",
    "    import shutil as _sh\n",
    "    _sh.rmtree(CACHE, ignore_errors=True)   # drop the downloaded source (saves ~25-31 GB)\n",
    "    print('freed source cache ->', CACHE)\n",
])

C_DEDUP = code([
    "# Cell 6 - the dedup passes (self + vs previous runs' lightweight fingerprints)\n",
    "# See dev_scripts/dedup_fire_scratch.py for the exact order and reasons.\n",
    "import os, subprocess, sys\n",
    "need(os.path.exists(RAW + '/data.yaml'), 'run Cell 5 (build raw pool) first')\n",
    "need(os.path.exists(UPLOAD + '/scripts/dedup_fire_scratch.py'), 'run Cell 4 (unpack bundle) first')\n",
    "\n",
    "with drive_log('06_dedup'):\n",
    "    os.makedirs(FPS_TRAIN, exist_ok=True)\n",
    "    os.makedirs(FPS_TEST, exist_ok=True)\n",
    "    d = UPLOAD + '/scripts/dedup_fire_scratch.py'\n",
    "    subprocess.run([sys.executable, d,\n",
    "                    '--pool', RAW, '--out', CLEAN,\n",
    "                    '--hamming', str(DEDUP_HAMMING),\n",
    "                    '--train-scope', DEDUP_TRAIN_SCOPE,\n",
    "                    '--max-bg-share', str(MAX_BG_SHARE),\n",
    "                    '--prev-train-index', PREV_TRAIN,\n",
    "                    '--prev-test-index', PREV_TEST,\n",
    "                    '--run-name', NAME,\n",
    "                    '--report', CLEAN + '_report',\n",
    "                    '--skip-broken', '--broken-out', '/content/broken'], check=True)\n",
    "    print(open(CLEAN + '_report/summary.txt', encoding='utf-8').read())\n",
    "    import shutil as _sh\n",
    "    _sh.rmtree(RAW, ignore_errors=True)     # clean pool is built; drop the raw pool\n",
    "    print('freed raw pool ->', RAW)\n",
])

C_VERIFY = code([
    "# Cell 7 - verify the CLEAN pool (read-only): splits, class coverage, integrity\n",
    "import collections, os\n",
    "need(os.path.exists(CLEAN + '/data.yaml'), 'run Cell 6 (dedup) first')\n",
    "\n",
    "EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')\n",
    "with drive_log('07_verify'):\n",
    "    for split in ('train', 'val', 'test'):\n",
    "        idir = os.path.join(CLEAN, split, 'images')\n",
    "        ldir = os.path.join(CLEAN, split, 'labels')\n",
    "        if not os.path.isdir(idir):\n",
    "            print('[%s] (absent)' % split)\n",
    "            continue\n",
    "        imgs = [f for f in os.listdir(idir) if f.lower().endswith(EXT)]\n",
    "        cls = collections.Counter()\n",
    "        missing = empty = 0\n",
    "        for f in imgs:\n",
    "            lb = os.path.join(ldir, os.path.splitext(f)[0] + '.txt')\n",
    "            if not os.path.isfile(lb):\n",
    "                missing += 1\n",
    "                continue\n",
    "            if os.path.getsize(lb) == 0:\n",
    "                empty += 1\n",
    "            for row in open(lb, encoding='utf-8'):\n",
    "                parts = row.split()\n",
    "                if parts:\n",
    "                    cls[int(float(parts[0]))] += 1\n",
    "        names = {i: n for i, n in enumerate(CLASSES)}\n",
    "        print('[%s] images=%d missing_label=%d background=%d boxes=%s'\n",
    "              % (split, len(imgs), missing, empty,\n",
    "                 ', '.join('%s=%d' % (names.get(k, k), v) for k, v in sorted(cls.items())) or '-'))\n",
    "    # fire vs not-fire balance of the CLEAN train split\n",
    "    _ti, _tl = os.path.join(CLEAN, 'train', 'images'), os.path.join(CLEAN, 'train', 'labels')\n",
    "    _npos = _nbg = 0\n",
    "    for _f in os.listdir(_ti):\n",
    "        _lb = os.path.join(_tl, os.path.splitext(_f)[0] + '.txt')\n",
    "        if os.path.isfile(_lb) and os.path.getsize(_lb) > 0:\n",
    "            _npos += 1\n",
    "        else:\n",
    "            _nbg += 1\n",
    "    print('\\ntrain balance: fire-positive=%d (%.0f%%)  background=%d (%.0f%%)  cap=%.0f%%'\n",
    "          % (_npos, 100 * _npos / max(1, _npos + _nbg),\n",
    "             _nbg, 100 * _nbg / max(1, _npos + _nbg), 100 * MAX_BG_SHARE))\n",
    "    print('\\ndata.yaml:')\n",
    "    print(open(CLEAN + '/data.yaml', encoding='utf-8').read().strip())\n",
    "    import shutil as _sh\n",
    "    print('\\n/content free: %.1f GB' % (_sh.disk_usage('/content').free / 1e9))\n",
])

C_TRAINER_CHECK = code([
    "# Cell 8 - sanity-check the detached trainer that ships in the bundle\n",
    "import os, subprocess, sys\n",
    "need(os.path.exists(UPLOAD + '/scripts/colab_train_scratch.py'), 'run Cell 4 (unpack bundle) first')\n",
    "trainer = UPLOAD + '/scripts/colab_train_scratch.py'\n",
    "print(subprocess.run([sys.executable, trainer, '--help'],\n",
    "                     capture_output=True, text=True).stdout[:900])\n",
])

C_LAUNCH = code([
    "# Cell 9 - LAUNCH training DETACHED (closing VS Code will NOT stop it)\n",
    "import hashlib, os, subprocess, sys\n",
    "need(os.path.exists(CLEAN + '/data.yaml'), 'run Cell 6 (dedup) first')\n",
    "need(os.path.exists(UPLOAD + '/scripts/colab_train_scratch.py'), 'run Cell 4 (unpack bundle) first')\n",
    "\n",
    "trainer = UPLOAD + '/scripts/colab_train_scratch.py'   # defined here too: Cell 8 is optional\n",
    "assert os.path.exists(trainer), 'trainer missing from the bundle - see Cell 4'\n",
    "\n",
    "# pre-flight: verify the Drive base model is the EXACT run-1 winner (avoid fine-tuning the wrong model)\n",
    "_base_md5 = hashlib.md5(open(BASE_MODEL, 'rb').read()).hexdigest()\n",
    "assert _base_md5 == EXPECTED_BASE_MD5, ('BASE_MODEL md5 MISMATCH: expected %s, got %s '\n",
    "                                        '- do NOT train on the wrong model' % (EXPECTED_BASE_MD5, _base_md5))\n",
    "print('base model md5 OK:', _base_md5)\n",
    "\n",
    "if os.path.exists(PIDFILE):\n",
    "    _old = int(open(PIDFILE).read().strip() or 0)\n",
    "    try:\n",
    "        os.kill(_old, 0)\n",
    "        raise SystemExit('a trainer is already running (pid %d). Watch it with Cell 10, '\n",
    "                         'or kill it first: !kill -9 %d' % (_old, _old))\n",
    "    except OSError:\n",
    "        pass\n",
    "\n",
    "os.makedirs(os.path.join(RUNS, NAME), exist_ok=True)\n",
    "args = [sys.executable, '-u', trainer,\n",
    "        '--data', CLEAN + '/data.yaml', '--model', BASE_MODEL,\n",
    "        '--runs', RUNS, '--name', NAME, '--mode', RUNS_MODE, '--local', LOCAL_RUNS,\n",
    "        '--log', LOG_LOCAL,\n",
    "        '--epochs', str(EPOCHS), '--batch', str(BATCH), '--imgsz', str(IMGSZ),\n",
    "        '--freeze', str(FREEZE), '--lr0', str(LR0), '--patience', str(PATIENCE),\n",
    "        '--save-period', str(SAVE_PERIOD)]\n",
    "with open(LOG_LOCAL, 'ab') as lf:\n",
    "    proc = subprocess.Popen(args, cwd='/content', stdout=lf,\n",
    "                            stderr=subprocess.STDOUT, start_new_session=True)\n",
    "open(PIDFILE, 'w').write(str(proc.pid))\n",
    "print('launched pid %d (own session -> survives losing the editor/kernel)' % proc.pid)\n",
    "print('local log :', LOG_LOCAL)\n",
    "print('Drive run :', os.path.join(RUNS, NAME), '(mirrored every epoch)')\n",
    "print('next      : run Cell 10 to watch progress')\n",
])

C_STATUS = code([
    "# Cell 10 - STATUS: re-attach to the detached run (no stdout stream needed)\n",
    "import os, subprocess, time\n",
    "\n",
    "with drive_log('10_status'):\n",
    "    need(os.path.exists(PIDFILE) or\n",
    "         os.path.exists(os.path.join(LOCAL_RUNS, NAME, 'results.csv')) or\n",
    "         os.path.exists(os.path.join(RUNS, NAME, 'results.csv')),\n",
    "         'no training launched - run Cell 9 (LAUNCH) first')\n",
    "    st, done, tail = training_status()\n",
    "    print('training status: %s (%d/%d epochs)' % (st, done, EPOCHS))\n",
    "    if st == 'failed':\n",
    "        print('-- log tail (failure) --')\n",
    "        print(tail)\n",
    "\n",
    "    pid = int(open(PIDFILE).read().strip() or 0) if os.path.exists(PIDFILE) else None\n",
    "\n",
    "    def proc_state(p):\n",
    "        if not p:\n",
    "            return 'no pid file'\n",
    "        try:\n",
    "            with open('/proc/%d/stat' % p) as fh:\n",
    "                fields = fh.read().split()\n",
    "            state = fields[2]\n",
    "        except (FileNotFoundError, IndexError):\n",
    "            return 'not running (no such process)'\n",
    "        if state == 'Z':\n",
    "            return 'ZOMBIE (exited - read the log tail below)'\n",
    "        return {'R': 'RUNNING', 'S': 'RUNNING (sleeping)', 'D': 'RUNNING (disk-sleep)',\n",
    "                'T': 'stopped'}.get(state, state)\n",
    "\n",
    "    print('trainer pid %s -> %s' % (pid, proc_state(pid)))\n",
    "    log_age = '%.0fs' % (time.time() - os.path.getmtime(LOG_LOCAL)) if os.path.exists(LOG_LOCAL) else 'n/a'\n",
    "    print('local log : %s (%d B, written %s ago)' % (LOG_LOCAL,\n",
    "          os.path.getsize(LOG_LOCAL) if os.path.exists(LOG_LOCAL) else 0, log_age))\n",
    "    run_local = os.path.join(LOCAL_RUNS, NAME)\n",
    "    run_drive = os.path.join(RUNS, NAME)\n",
    "\n",
    "    for label, d in (('local', run_local), ('Drive', run_drive)):\n",
    "        print('\\n[%s] %s' % (label, d))\n",
    "        if not os.path.isdir(d):\n",
    "            print('   (missing)')\n",
    "            continue\n",
    "        csvp = os.path.join(d, 'results.csv')\n",
    "        if os.path.exists(csvp):\n",
    "            rows = [r for r in open(csvp).read().splitlines() if r.strip()]\n",
    "            print('   results.csv  epochs=%d' % max(0, len(rows) - 1))\n",
    "            if len(rows) > 1:\n",
    "                print('   last row    :', rows[-1][:140])\n",
    "        w = os.path.join(d, 'weights')\n",
    "        if os.path.isdir(w):\n",
    "            for f in sorted(os.listdir(w)):\n",
    "                p = os.path.join(w, f)\n",
    "                print('   %-9s %7.1f MB' % (f, os.path.getsize(p) / 1e6))\n",
    "\n",
    "    print('\\n-- local log tail --')\n",
    "    if os.path.exists(LOG_LOCAL):\n",
    "        print('\\n'.join(open(LOG_LOCAL, errors='replace').read().splitlines()[-15:]))\n",
    "    else:\n",
    "        print('(no log yet)')\n",
    "\n",
    "    print('\\n-- GPU --')\n",
    "    print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',\n",
    "                          '--format=csv,noheader'],\n",
    "                         capture_output=True, text=True).stdout.strip() or '(n/a)')\n",
])

C_WAIT = code([
    "# Cell 11 - WAIT for the run to finish (OPTIONAL; safe to interrupt)\n",
    "import os, time\n",
    "\n",
    "WAIT_POLL_S  = 60      # seconds between polls\n",
    "WAIT_MAX_MIN = 480     # stop waiting after this long (training keeps running)\n",
    "\n",
    "run_local = os.path.join(LOCAL_RUNS, NAME)\n",
    "run_drive = os.path.join(RUNS, NAME)\n",
    "started = (os.path.exists(PIDFILE) or\n",
    "           os.path.exists(os.path.join(run_local, 'results.csv')) or\n",
    "           os.path.exists(os.path.join(run_drive, 'results.csv')))\n",
    "need(started, 'no training launched - run Cell 9 (LAUNCH) first')\n",
    "\n",
    "t0 = time.time(); last = -1\n",
    "while True:\n",
    "    st, n, tail = training_status()\n",
    "    if n != last:\n",
    "        print('[%s] %s %d/%d epochs | waiting %d min'\n",
    "              % (time.strftime('%H:%M:%S'), st, n, EPOCHS,\n",
    "                 int((time.time() - t0) / 60)), flush=True)\n",
    "        last = n\n",
    "    if st in ('completed', 'early-stopped'):\n",
    "        print('-> training finished successfully (%s).' % st)\n",
    "        break\n",
    "    if st == 'failed':\n",
    "        print('-> training FAILED. log tail:')\n",
    "        print(tail)\n",
    "        break\n",
    "    if st in ('none', 'unknown'):\n",
    "        print('-> no live trainer (%s). Check Cell 10, then resume with Cell 12.' % st)\n",
    "        break\n",
    "    if (time.time() - t0) > WAIT_MAX_MIN * 60:\n",
    "        print('WAIT_MAX_MIN reached - no longer waiting (training continues detached)')\n",
    "        break\n",
    "    time.sleep(WAIT_POLL_S)\n",
])

C_RESUME = code([
    "# Cell 12 - RESUME an interrupted run (detached too). Refuses if the run FINISHED.\n",
    "import os, shutil, subprocess, sys, torch\n",
    "need(os.path.exists(UPLOAD + '/scripts/colab_train_scratch.py'), 'run Cell 4 (unpack bundle) first')\n",
    "\n",
    "trainer = UPLOAD + '/scripts/colab_train_scratch.py'\n",
    "run_local, run_drive = os.path.join(LOCAL_RUNS, NAME), os.path.join(RUNS, NAME)\n",
    "if not os.path.isdir(run_local) and os.path.isdir(run_drive):\n",
    "    shutil.copytree(run_drive, run_local, dirs_exist_ok=True)\n",
    "    print('restored run dir from Drive ->', run_local)\n",
    "\n",
    "ck = os.path.join(run_local, 'weights', 'last.pt')\n",
    "if not os.path.exists(ck):\n",
    "    ck = os.path.join(run_drive, 'weights', 'last.pt')\n",
    "print('checkpoint:', ck)\n",
    "assert os.path.exists(ck), 'no last.pt - run Cell 9 first'\n",
    "\n",
    "meta = torch.load(ck, map_location='cpu', weights_only=False)\n",
    "ta = meta.get('train_args') or {}\n",
    "done, total = int(meta.get('epoch', -1)) + 1, int(ta.get('epochs', 0) or 0)\n",
    "resumable = meta.get('optimizer') is not None\n",
    "print('epoch %d/%d - optimizer %s' % (done, total, 'present' if resumable else 'STRIPPED (finished)'))\n",
    "if not resumable:\n",
    "    print('=> nothing to resume: this run is COMPLETE. Use Cell 13 to collect best.pt.')\n",
    "else:\n",
    "    with open(LOG_LOCAL, 'ab') as lf:\n",
    "        proc = subprocess.Popen([sys.executable, '-u', trainer, '--resume', ck],\n",
    "                                cwd='/content', stdout=lf, stderr=subprocess.STDOUT,\n",
    "                                start_new_session=True)\n",
    "    open(PIDFILE, 'w').write(str(proc.pid))\n",
    "    print('resumed detached, pid', proc.pid, '-> watch with Cell 10')\n",
])

C_COLLECT = code([
    "# Cell 13 - COLLECT + EVALUATE on the held-out test split + compare vs previous best\n",
    "# (BLOCKS until the detached trainer launched by Cell 9 has finished)\n",
    "import hashlib, json, os, shutil, time\n",
    "need(os.path.exists(CLEAN + '/data.yaml'), 'run Cell 6 (dedup) first')\n",
    "from ultralytics import YOLO\n",
    "\n",
    "run_local = os.path.join(LOCAL_RUNS, NAME)\n",
    "run_drive = os.path.join(RUNS, NAME)\n",
    "have_ckpt = (os.path.exists(os.path.join(run_local, 'weights', 'best.pt')) or\n",
    "             os.path.exists(os.path.join(run_drive, 'weights', 'best.pt')) or\n",
    "             os.path.exists(os.path.join(run_local, 'weights', 'last.pt')) or\n",
    "             os.path.exists(os.path.join(run_drive, 'weights', 'last.pt')))\n",
    "need(os.path.exists(PIDFILE) or\n",
    "     os.path.exists(os.path.join(run_local, 'results.csv')) or\n",
    "     os.path.exists(os.path.join(run_drive, 'results.csv')) or have_ckpt,\n",
    "     'run Cell 9 (LAUNCH) first')\n",
    "\n",
    "# 1) WAIT for the detached trainer to finish (Cell 9 returns immediately)\n",
    "WAIT_POLL_S = 30\n",
    "WAIT_MAX_MIN = 480\n",
    "def _trainer_state():\n",
    "    pid = int(open(PIDFILE).read().strip() or 0) if os.path.exists(PIDFILE) else 0\n",
    "    if pid and os.path.exists('/proc/%d' % pid):\n",
    "        try:\n",
    "            s = open('/proc/%d/stat' % pid).read().split()[2]\n",
    "        except (FileNotFoundError, IndexError):\n",
    "            s = 'Z'\n",
    "        return 'running' if s != 'Z' else 'zombie'\n",
    "    return 'gone' if pid else 'none'\n",
    "\n",
    "t0 = time.time(); n = 0; st = 'none'\n",
    "while True:\n",
    "    rows = []\n",
    "    for p in (os.path.join(run_local, 'results.csv'), os.path.join(run_drive, 'results.csv')):\n",
    "        if os.path.exists(p):\n",
    "            rows = [r for r in open(p, errors='replace').read().splitlines() if r.strip()]\n",
    "            break\n",
    "    n = max(0, len(rows) - 1)\n",
    "    st = _trainer_state()\n",
    "    if st == 'none' or n >= EPOCHS or st in ('gone', 'zombie'):\n",
    "        break\n",
    "    if (time.time() - t0) > WAIT_MAX_MIN * 60:\n",
    "        print('WAIT_MAX_MIN reached - collecting what exists (training may still be running)')\n",
    "        break\n",
    "    print('[%s] waiting for training ... %d/%d epochs (trainer %s)'\n",
    "          % (time.strftime('%H:%M:%S'), n, EPOCHS, st), flush=True)\n",
    "    time.sleep(WAIT_POLL_S)\n",
    "print('training finished: %d epoch row(s), trainer %s' % (n, st))\n",
    "_st, _n, _tail = training_status()\n",
    "if _st == 'failed':\n",
    "    print('\\nTRAINING FAILED - log tail:')\n",
    "    print(_tail)\n",
    "    raise SystemExit('training failed - fix the issue, then resume (Cell 12) or re-launch (Cell 9)')\n",
    "if _st in ('completed', 'early-stopped'):\n",
    "    print('training %s - collecting the best checkpoint.' % _st)\n",
    "\n",
    "# 2) locate the best checkpoint (local first, then Drive)\n",
    "src = None\n",
    "for cand in (os.path.join(LOCAL_RUNS, NAME, 'weights', 'best.pt'),\n",
    "             os.path.join(RUNS, NAME, 'weights', 'best.pt'),\n",
    "             os.path.join(LOCAL_RUNS, NAME, 'weights', 'last.pt'),\n",
    "             os.path.join(RUNS, NAME, 'weights', 'last.pt')):\n",
    "    if os.path.exists(cand):\n",
    "        src = cand\n",
    "        break\n",
    "assert src, 'no checkpoint found - did Cell 9 run?'\n",
    "out_pt = os.path.join(DRIVE_DIR, VERSION, NAME + '.pt')\n",
    "shutil.copy2(src, out_pt)\n",
    "md5 = hashlib.md5(open(out_pt, 'rb').read()).hexdigest()\n",
    "print('candidate ->', out_pt)\n",
    "print('md5: %s | bytes: %d' % (md5, os.path.getsize(out_pt)))\n",
    "\n",
    "# 3) evaluate on the held-out test split (val points at test/images)\n",
    "test_yaml = CLEAN + '/data_test.yaml'\n",
    "lines = [ln for ln in open(CLEAN + '/data.yaml', encoding='utf-8') if not ln.startswith('val:')]\n",
    "lines.append('val: test/images\\n')\n",
    "open(test_yaml, 'w', encoding='utf-8').writelines(lines)\n",
    "m = YOLO(out_pt)\n",
    "res = m.val(data=test_yaml, imgsz=IMGSZ, batch=BATCH, device=0, verbose=False)\n",
    "metrics = {\n",
    "    'map50': round(float(res.box.map50), 4),\n",
    "    'map50_95': round(float(res.box.map), 4),\n",
    "    'precision': round(float(res.box.mp), 4),\n",
    "    'recall': round(float(res.box.mr), 4),\n",
    "    'per_class_map50': [round(float(x), 4) for x in (res.box.maps or [])],\n",
    "}\n",
    "print('classes:', list(m.names.values()))\n",
    "print('test metrics:', metrics)\n",
    "\n",
    "# 4) registry: append this run, then compare vs the previous best on mAP@50\n",
    "entry = {\n",
    "    'name': NAME,\n",
    "    'date': time.strftime('%Y-%m-%d %H:%M:%S UTC'),\n",
    "    'classes': CLASSES,\n",
    "    'base_model': BASE_MODEL,\n",
    "    'md5': md5,\n",
    "    'bytes': os.path.getsize(out_pt),\n",
    "    'metrics': metrics,\n",
    "    'metric_key': 'map50',\n",
    "}\n",
    "registry = []\n",
    "if os.path.exists(REGISTRY):\n",
    "    registry = json.load(open(REGISTRY, encoding='utf-8'))\n",
    "elif os.path.exists(PREV_REG):\n",
    "    registry = json.load(open(PREV_REG, encoding='utf-8'))   # chain the previous run's history\n",
    "registry.append(entry)\n",
    "json.dump(registry, open(REGISTRY, 'w', encoding='utf-8'), indent=2)\n",
    "\n",
    "prev = registry[:-1]\n",
    "best_prev = max(prev, key=lambda e: e['metrics']['map50']) if prev else None\n",
    "cur = metrics['map50']\n",
    "if best_prev is None:\n",
    "    decision = 'FIRST RUN - promoted'\n",
    "    active = entry\n",
    "elif cur > best_prev['metrics']['map50']:\n",
    "    decision = 'PROMOTED (map50 %.4f > previous best %.4f)' % (cur, best_prev['metrics']['map50'])\n",
    "    active = entry\n",
    "else:\n",
    "    decision = 'REGRESSED (map50 %.4f <= previous best %.4f) - previous best stays ACTIVE' % (cur, best_prev['metrics']['map50'])\n",
    "    active = best_prev\n",
    "json.dump({'active': active['name']}, open(DRIVE_DIR + '/active.json', 'w', encoding='utf-8'), indent=2)\n",
    "print('\\nDECISION:', decision)\n",
    "print('ACTIVE  :', active['name'])\n",
    "if best_prev is not None and cur <= best_prev['metrics']['map50']:\n",
    "    print('\\nREGRESSION - decide (human call, nothing is silently overwritten):')\n",
    "    print('  (a) RETRAIN the previous version - edit NAME + hyperparameters and run Cells 5-9 again, or')\n",
    "    print('  (b) CONTINUE with this regressed one - if its failure mode (e.g. fewer false positives)')\n",
    "    print('      is what you need despite the lower mAP, keep ' + NAME + ' as the base for the next run.')\n",
    "\n",
    "# 5) promote THIS run's lightweight fingerprints to the persistent Drive index\n",
    "#    (only now - training succeeded - so the index accumulates exactly the trained runs)\n",
    "for f, dst_dir in (('run_train_index.jsonl', FPS_TRAIN), ('run_test_index.jsonl', FPS_TEST)):\n",
    "    p = CLEAN + '_report/' + f\n",
    "    if os.path.exists(p):\n",
    "        shutil.copy2(p, os.path.join(dst_dir, NAME + '.jsonl'))\n",
    "        print('fingerprint index updated:', os.path.join(dst_dir, NAME + '.jsonl'))\n",
    "print('\\nregistry ->', REGISTRY)\n",
])

C_NEXT = md([
    "## Next steps\n",
    "\n",
    "1. Pull the candidate `DRIVE_DIR/VERSION/<NAME>.pt` from Drive back into the repo and judge it\n",
    "   **locally** against the full matrix (`dev_scripts/compare_fire_models.py`) and the on-domain\n",
    "   CCTV audit (`dev_scripts/test_fire_model.py` on the held-out evidence / `domain_test`).\n",
    "2. If it is a genuine win, convert + archive it per\n",
    "   [`plans/model-versioning.md`](../plans/model-versioning.md):\n",
    "   `prep_fire_model.sh` (writes `labelmap.txt` = your `CLASSES`) -> `promote_fire_model.sh` ->\n",
    "   on-host `firewatch.py --dry-run` pilot (the real acceptance test).\n",
    "3. For the **next run**, bump `VERSION` (e.g. `v3`) and `PREV_VER` (e.g. `v2`); the new set is\n",
    "   automatically deduped against every previous run via its `PREV_*` fingerprint dirs.\n",
    "4. Per-stage logs live on Drive at `DRIVE_DIR/VERSION/logs/`.\n",
])


def version_cell():
    """Markdown cell stamped with the notebook version, build time and git hash."""
    import datetime
    import subprocess
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or "n/a"
    except Exception:
        git = "n/a"
    return md([
        "> **Notebook version v%s** · built %s · git `%s`\n" % (NOTEBOOK_VERSION, ts, git),
    ])


def build(out=DEFAULT_OUT):
    cells = _demagic([
        version_cell(),
        C_TITLE, C_SETUP, C_CONFIG, C_LOGGING, C_BUNDLE, C_SOURCES, C_DEDUP, C_VERIFY,
        C_TRAINER_CHECK, C_LAUNCH, C_STATUS, C_WAIT, C_RESUME, C_COLLECT, C_NEXT,
    ])
    _check_no_magics(cells)
    _check_cross_cell_names(cells)
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "version": NOTEBOOK_VERSION},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print("wrote %s (%d cells)" % (out, len(cells)))


def _demagic(cells):
    """Glue any continuation line that begins with '%' back onto the previous line."""
    for c in cells:
        if c["cell_type"] != "code":
            continue
        out = []
        for ln in "".join(c["source"]).split("\n"):
            if ln.lstrip().startswith("%") and out and out[-1].strip():
                out[-1] = out[-1].rstrip() + " " + ln.lstrip()
            else:
                out.append(ln)
        c["source"] = "\n".join(out)
    return cells


def _check_no_magics(cells):
    """Refuse to write a notebook whose code cells could be mis-read as magics."""
    bad = [(i, ln.strip()[:60])
           for i, c in enumerate(cells, 1) if c["cell_type"] == "code"
           for ln in "".join(c["source"]).splitlines()
           if ln.lstrip().startswith("%")]
    if bad:
        raise SystemExit("refusing to write: '%%' at line start could be read as a magic: %r"
                         % bad[:3])


def _check_cross_cell_names(cells):
    """Refuse a notebook where a cell READS a name no EARLIER cell defines."""
    import ast
    import builtins

    known = set(dir(builtins))
    problems = []
    for i, c in enumerate(cells, 1):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue                                    # reported by the caller's compile pass
        stores, loads = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                (stores if isinstance(node.ctx, ast.Store) else loads).add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    stores.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stores.add(node.name)
            elif isinstance(node, ast.arg):
                stores.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                stores.add(node.name)
            elif isinstance(node, ast.Global):
                stores.update(node.names)
        missing = sorted(n for n in loads if n not in known and n not in stores)
        if missing:
            problems.append((i, missing))
        known |= stores
    if problems:
        raise SystemExit("refusing to write: a cell reads names no earlier cell defines "
                         "(cell, names): %r" % (problems[:3],))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="notebook output path (default: %(default)s)")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
