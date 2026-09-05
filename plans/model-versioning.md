# Versioned fire/smoke model checkpoints — `models/fire/versions/`

**Status:** ✅ designed + implemented (2026-09-05).

**Goal:** give the growing set of fire/smoke YOLO checkpoints (currently two, *both*
incidentally named `best.pt`) a durable, self-contained, versioned home with an explicit
**naming convention**, so every candidate is identifiable, comparable and promotable
without ever clobbering the active production model.

**Why:** the active checkpoint [`models/fire/best.pt`](../models/fire/best.pt) (HF YOLO26-S)
and the domain-adapted candidate [`dataset/abonia_eval/finetune/best_finetuned_abonia.pt`](../dataset/abonia_eval/finetune/best_finetuned_abonia.pt)
(fine-tuned on ~877 extra Abonia `fire-8` train images) are both large, both git-ignored,
and both conventionally named `best.pt`. As more fine-tunes/pilots arrive, an
undifferentiated pile of `best.pt` files becomes impossible to reason about. This plan
stores each model as an immutable, self-describing version folder, and keeps a *copy* of
the chosen one at the `models/fire/` root as the **active set** that `firewatch` actually
loads (contract unchanged).

---

## 1. Current artifacts (pre-change)

| File | Role | Size | md5 | Class order | Training |
|---|---|---|---|---|---|
| [`models/fire/best.pt`](../models/fire/best.pt) | **active** source `.pt` (root) | 20.3 MB | `2fd972183c2ffec0d327ec534c119086` | `fire`(0) `other`(1) `smoke`(2) | HF `SalahALHaismawi/yolov26-fire-detection`, 8,939 imgs / 100 ep @ 640 |
| [`dataset/abonia_eval/finetune/best_finetuned_abonia.pt`](../dataset/abonia_eval/finetune/best_finetuned_abonia.pt) | candidate (eval workdir) | 20.3 MB | `ddc0cd6c3089c0da140017f881673327` | `fire`(0) `other`(1) `smoke`(2) | fine-tune from v1 on Abonia `fire-8` **train** (877 imgs), 5 ep @ 640, freeze 10, lr0 0.001 (CPU smoke run) |
| `best.xml` / `best.bin` / `labelmap.txt` | **active** OpenVINO IR (host-only today) | — | — | must match model order | deployed on `ssh.mazr3a.garden`; **not present locally** |

> Class order is **`fire / other / smoke`** (verified on both checkpoints) — note this
> differs from what `models/fire/README.md` historically claimed (`fire/smoke/other`).
> Production is fire-only (index 0) so alerts are unaffected; keep `labelmap.txt` in model
> order when regenerating IR.

---

## 2. Naming convention

Each version lives in a directory named:

```
v<N>-<YYYY-MM-DD>-<slug>
```

- `v<N>` — monotonically increasing version number (1, 2, 3, …), never reused. Higher =
  newer intent.
- `<YYYY-MM-DD>` — ISO creation/fine-tune date.
- `<slug>` — short self-description: `<source>/<family>` + `<training-signature>`, e.g.
  `hf-yolo26s-8939img` (pretrained HF YOLO26-S, its 8,939-image pool) or
  `hf-abonia877-ft5ep` (HF base fine-tuned on Abonia `fire-8` train, 877 imgs, 5 epochs).

Concrete today:

| Version dir | Checkpoint inside | Meaning |
|---|---|---|
| `v1-2026-09-05-hf-yolo26s-8939img` | `model.pt` | HF YOLO26-S baseline (unchanged, still **active**) |
| `v2-2026-09-05-hf-abonia877-ft5ep` | `model.pt` | Abonia fine-tune candidate (not promoted) |

Inside each dir the checkpoint is a neutral **`model.pt`** — never `best.pt` — so a
version folder can never be mistaken for "the active one". Only the `models/fire/` root
may contain `best.*`.

Authoritative structured provenance lives in each folder's **`VERSION.json`**; the
human-readable summary of every version lives in the tracked
[`models/fire/VERSIONS.md`](../models/fire/VERSIONS.md).

---

## 3. Directory layout

```
models/fire/
├── best.pt                # ACTIVE source checkpoint (copy of active version's model.pt)
├── best.xml               # ACTIVE OpenVINO IR graph (what firewatch loads - host)
├── best.bin               # ACTIVE OpenVINO IR weights
├── labelmap.txt           # ACTIVE class list, index order = model class order
├── VERSIONS.md            # tracked registry/index of all versions (this is the durable record)
├── README.md              # tracked
└── versions/              # GIT-IGNORED archive: one immutable dir per version
    ├── v1-2026-09-05-hf-yolo26s-8939img/
    │   ├── model.pt       # HF YOLO26-S baseline checkpoint (copy)
    │   └── VERSION.json   # provenance manifest
    └── v2-2026-09-05-hf-abonia877-ft5ep/
        ├── model.pt       # Abonia fire-8 fine-tune checkpoint (copy)
        └── VERSION.json
```

- `versions/` is **git-ignored** (large `.pt` + machine manifests). The *tracked* index is
  [`models/fire/VERSIONS.md`](../models/fire/VERSIONS.md) — the durable, reviewable record.
- Version folders are **immutable by convention**: to iterate, add `v<N+1>`; never edit an
  existing one in place (if you must supersede an active version, promote the new one).

---

## 4. `VERSION.json` schema

```json
{
  "id": "v1-2026-09-05-hf-yolo26s-8939img",
  "version": 1,
  "date": "2026-09-05",
  "checkpoint": "model.pt",
  "arch": "yolo26s",
  "classes": ["fire", "other", "smoke"],
  "input_size": 640,
  "source": { "type": "huggingface", "repo": "SalahALHaismawi/yolov26-fire-detection" },
  "training": { "dataset": "...", "images": 8939, "epochs": 100 },
  "metrics": { "map50": 0.949, "map50_95": 0.68, "precision": 0.896, "recall": 0.888, "note": "author-reported on own validation" },
  "license": "MIT",
  "parent": null,
  "status": "active",
  "md5": "...",
  "size_bytes": 20301317
}
```

`parent` links a fine-tune to the version it started from (v2 → `v1-...`). `status` is one
of `active` / `candidate` / `superseded` and is updated (manually or on promote) together
with [`models/fire/VERSIONS.md`](../models/fire/VERSIONS.md).

---

## 5. Active-set semantics (contract unchanged)

- **The root `models/fire/` is the "active set"** — `best.pt` (source) plus
  `best.xml`/`best.bin`/`labelmap.txt` (OpenVINO IR) are whatever `firewatch` uses
  (`MODEL_DIR=/models/fire`, container hardcodes `best.xml`). This is **not** changed by
  this plan.
- Promoting a version = copying its artifacts to the root (see §6). Nothing in
  `scripts/firewatch.py` or `docker-compose.yml` changes.
- The root `best.pt` is just a copy of the active version's `model.pt` — the canonical
  source of truth for each model remains its `versions/` folder.

---

## 6. Promote workflow — `scripts/promote_fire_model.sh`

```bash
./scripts/promote_fire_model.sh v2-2026-09-05-hf-abonia877-ft5ep
```

What it does:
1. Resolves the version dir under `models/fire/versions/` (exact name or unique prefix).
2. Copies its `model.pt` → `models/fire/best.pt` (the active source checkpoint).
3. If the version dir also carries a ready OpenVINO IR (`best.xml` + `best.bin` +
   `labelmap.txt`) it copies all three to the root (deploy-ready).
4. If **no** IR is present (the common case, since IR is generated from the promoted
   `.pt`), it prints the exact next command —
   `./scripts/prep_fire_model.sh models/fire/best.pt 640 "fire,other,smoke"` —
   and warns that the running container still uses the *previous* IR until one is
   regenerated + deployed.
5. Reminds you to flip `status` in `VERSION.json`/`VERSIONS.md` and to run
   `./scripts/deploy_firewatch.sh` (push to host) + `--check`/`--dry-run` (per
   `.roo/rules/sshuser.md`).

**Deploy is now conditional-safe:** [`scripts/deploy_firewatch.sh`](../scripts/deploy_firewatch.sh)
was hardened (§8 below) to push **only the ACTIVE model files** (`best.xml` + `best.bin` +
`labelmap.txt` [+ `best.pt`]) into the host's `models/fire/` — never the whole local
`models/` tree, so the versioned `versions/` archive stays local and a working host model is
never erased by a directory replace. Each file is pushed only when its md5 differs from the
host (atomic `.new` → `mv`); if the local ACTIVE OpenVINO IR is incomplete but the host
already runs one, the host model is left untouched and only code/config deploy. `frigate`/
`mqtt` are never restarted — only the `firewatch` container, when its model/code/config
changed. To actually change the served model: generate the IR from the promoted `.pt`
([`scripts/prep_fire_model.sh`](../scripts/prep_fire_model.sh), which also writes
`labelmap.txt` in model order), then deploy.

---

## 7. Implementation checklist

- [x] Design + user decision (Option 1: versioned dirs + `VERSION.json` + root active set).
- [x] Verify both checkpoints (md5, size, class order `fire/other/smoke`).
- [x] Create `models/fire/versions/v1-*` with baseline `model.pt` + `VERSION.json`.
- [x] Create `models/fire/versions/v2-*` with fine-tuned `model.pt` + `VERSION.json`.
- [x] Add `scripts/promote_fire_model.sh`.
- [x] Ignore `models/fire/versions/` in `.gitignore`.
- [x] Write tracked `models/fire/VERSIONS.md` registry.
- [x] Update `models/fire/README.md` (layout, convention, promote usage, class-order note).
- [x] Harden `scripts/deploy_firewatch.sh`: conditional ACTIVE-model push (never whole
      `models/`, never erases a working host model), firewatch-only restart, `--check` +
      `--dry-run` verification.
- [x] Commit per logical group.

**No remote deploy for this step:** versioning only *archives* the existing checkpoints
and leaves the active `models/fire/` root untouched (v1 is still active; no OpenVINO IR is
regenerated or re-promoted). The remote host's running `firewatch` model is unchanged.

---

## 8. Implementation log (per `.roo/rules/Agents.md`)

- **2026-09-05 — design:** user chose Option 1 (versioned dirs + `VERSION.json` manifest,
  root `best.*` = active set, promote helper). Recorded here + in
  [`models/fire/VERSIONS.md`](../models/fire/VERSIONS.md).
- **2026-09-05 — verify:** read both checkpoints via Ultralytics → md5/size/class order
  table in §1 confirmed (`fire/other/smoke`).
- **2026-09-05 — archive:** created `models/fire/versions/{v1,v2}-*/`; copied
  `models/fire/best.pt` → `v1/.../model.pt` and
  `dataset/abonia_eval/finetune/best_finetuned_abonia.pt` → `v2/.../model.pt` (originals
  left in place — non-destructive); wrote each `VERSION.json`.
- **2026-09-05 — helper + docs:** added `scripts/promote_fire_model.sh`; `.gitignore` now
  ignores `models/fire/versions/`; wrote tracked `models/fire/VERSIONS.md`; updated
  `models/fire/README.md` with the layout, naming convention, promote usage and the
  `fire/other/smoke` class-order note.
- **2026-09-05 — commit:** `ba4954b` — "feat(models): versioned fire checkpoint archive +
  naming convention + promote helper" (implementation).
- **2026-09-05 — deploy hardening:** rewrote `scripts/deploy_firewatch.sh` per user request —
  model push is now conditional on an md5 diff and limited to the ACTIVE files (the
  `versions/` archive stays local; a host model is never erased); only the `firewatch`
  container is restarted; added `--dry-run` verification. Docs synced
  (`promote_fire_model.sh`, `models/fire/VERSIONS.md`).
- **2026-09-05 — commit:** `xxxxxxx` — "fix(deploy): conditional ACTIVE-model push" (filled
  after commit).
