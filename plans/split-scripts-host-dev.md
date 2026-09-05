# Split `scripts/` into host scripts (`scripts/`) + dev scripts (`dev_scripts/`)

## Goal

The `scripts/` folder mixes two different kinds of files:

- **Host-runtime scripts** — deployed to and run on the remote Frigate host
  (`/home/dr/frigate/scripts/`) or inside the `firewatch` container, or invoked
  from the host crontab.
- **Dev/local scripts** — run only in this workspace for development, dataset /
  model work, debugging, and local deploy orchestration. These must NOT ship to
  the host.

Decision (per user): keep the host scripts in `scripts/` and put every other
script in a new top-level `dev_scripts/` folder.

## Classification

### Host scripts — stay in `scripts/`

| File | Where it runs |
| --- | --- |
| `cleanup_media.sh` | host crontab (`/home/dr/frigate/scripts/`) |
| `collect_sensors.py` | host + firewatch container (`sys.path` import by the other host scripts) |
| `crontab.sample` | installed on the host via `crontab` |
| `diagnose_detection.py` | run on the host (`/home/dr/frigate`) |
| `firewatch.py` | firewatch container (`/scripts/firewatch.py` entrypoint) |
| `machine-monitor.py` | host crontab |
| `machine-status.py` | host crontab |
| `verify_remote.py` | deployed to host, run there after deploy |

Why `scripts/` keeps these: `docker-compose.yml` mounts `./scripts` → `/scripts:ro`
into the firewatch container, the `firewatch/Dockerfile` entrypoint is
`/scripts/firewatch.py`, deploy scripts install them into the host's `scripts/`
dir, and they share `import collect_sensors as lib` via `sys.path` relative to
their own directory (which would break if they were separated).

### Dev scripts — move to `dev_scripts/`

- `analyze_fire_dataset.py`
- `build_fire_eval_subset.py`
- `build_fire_large_colab_nb.py` (generates `notebooks/...`)
- `build_fire_negatives_eval.py`
- `deploy_config.sh` (local orchestrator — pushes config to host, never lives there)
- `deploy_firewatch.sh` (local orchestrator — bundles + pushes to host)
- `find_unlabelled_images.py`
- `prep_coco_model.sh`
- `prep_fire_large_dataset.py`
- `prep_fire_model.sh`
- `promote_fire_model.sh`
- `test_fire_model.py`

## Reference updates required

- `dev_scripts/deploy_config.sh`: `LOCAL_VERIFY` must become
  `${SCRIPT_DIR}/../scripts/verify_remote.py` (verify_remote.py stays in `scripts/`).
- `dev_scripts/deploy_firewatch.sh`: `ROOT_DIR` still resolves to the repo root
  (one level above `dev_scripts/`); it bundles `scripts/firewatch.py` +
  `scripts/collect_sensors.py`, which remain in `scripts/` — no path change needed,
  but double-checked.
- Docstrings inside moved dev scripts that reference other moved dev scripts
  (e.g. `build_fire_negatives_eval.py` → `test_fire_model.py`,
  `build_fire_large_colab_nb.py` → `prep_fire_large_dataset.py`).
- The generated notebook (`notebooks/fire-large-finetune-colab.ipynb`) and its
  generator markdown cells.
- Living docs: `README.md`, `models/fire/README.md`, `models/fire/VERSIONS.md`,
  `models/coco/README.md`, and `plans/*.md` links that point at moved scripts.

## Implementation record

- **Plan written** (this file), then executed:
- **Moved** (git mv) 12 dev files from `scripts/` to a new `dev_scripts/`:
  `analyze_fire_dataset.py`, `build_fire_eval_subset.py`, `build_fire_large_colab_nb.py`,
  `build_fire_negatives_eval.py`, `deploy_config.sh`, `deploy_firewatch.sh`,
  `find_unlabelled_images.py`, `prep_coco_model.sh`, `prep_fire_large_dataset.py`,
  `prep_fire_model.sh`, `promote_fire_model.sh`, `test_fire_model.py`.
- `scripts/` now holds ONLY the 8 host-runtime files (unchanged paths → on-host layout,
  `docker-compose.yml` `./scripts:/scripts:ro` mount, `Dockerfile` `/scripts/firewatch.py`
  entrypoint and host crontab paths all keep working).
- **Reference updates:**
  - `dev_scripts/deploy_config.sh`: `LOCAL_VERIFY` now `../scripts/verify_remote.py`
    (verify_remote.py stayed in `scripts/`).
  - `dev_scripts/deploy_firewatch.sh`: `ROOT_DIR` still resolves to repo root (one level
    above `dev_scripts/`); still bundles `scripts/firewatch.py` + `scripts/collect_sensors.py`.
  - Repo-wide path rewrite `scripts/<dev>` → `dev_scripts/<dev>` across tracked text files
    (all moved scripts' docstrings/help, generated notebook + its generator cells, and links
    in `README.md` (tree only), `models/fire/README.md`, `models/fire/VERSIONS.md`,
    `models/coco/README.md`, `plans/*.md`).
  - Regenerated `notebooks/fire-large-finetune-colab.ipynb` from its (updated) generator.
  - `.gitignore` comment and `firewatch/Dockerfile` comment (monitor_lib → collect_sensors).
- **Verification:** `python3 -m py_compile` on all `.py` in `scripts/` + `dev_scripts/`;
  `bash -n` on all `.sh`; `git grep` confirms no stale `scripts/<dev>` references remain.
- Commit created per .roo/rules/Agents.md.
- Deploy to host is NOT required by this change (host runtime files and on-host layout are
  unchanged) - verified via `deploy_firewatch.sh` / `crontab.sample` path analysis.
