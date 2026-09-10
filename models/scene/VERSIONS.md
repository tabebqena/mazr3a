# Scene-description model — version registry

Registry of the **SmolVLM2** OpenVINO export installed under this directory for
the `scenewatch` service. Unlike [`models/fire/VERSIONS.md`](../fire/VERSIONS.md)
there is no promote/archive helper: `models/scene/` is **git-ignored** and holds
exactly one ACTIVE export (see [`README.md`](README.md)). Re-running
[`dev_scripts/prep_scene_model.sh`](../../dev_scripts/prep_scene_model.sh)
replaces it.

**Rule:** after every `prep_scene_model.sh` run, append the line the script prints
(it computes the date, the source, the on-disk size and the md5 of the largest
`.bin`) and mark the superseded row `SUPERSEDED`.

| Date (UTC) | Source | How | Size | md5 (largest `.bin`) | Status |
|---|---|---|---|---|---|
| _none yet_ | `circulus/SmolVLM2-500M-ov-sym-int4` | download (`--repo int4`) | — | — | not fetched |

## Fields

| Column | Meaning |
|---|---|
| Date (UTC) | When the export was fetched/built |
| Source | The upstream model / repo id (`--repo`), i.e. what `prep_scene_model.sh` printed |
| How | `download` (pre-converted OV repo via curl) or `export` (`optimum-cli`) |
| Size | `du -sh models/scene/` |
| md5 | md5 of the largest `*.bin` (the language decoder weights — the bulk of the model) |
| Status | `ACTIVE` (currently in `models/scene/`) or `SUPERSEDED` |

## Notes

- **Size vs RAM.** The default `int4` export is ~356 MB on disk and roughly the
  same resident; the fp16 variants are several times larger. The host has ~7.5 GiB
  total shared with Frigate + firewatch + portal, so `int4` is the safe default.
- **Device.** CPU vs iGPU is a runtime choice (`MODEL_DEVICE` in
  [`config/scenewatch.conf`](../../config/scenewatch.conf)), not an export
  property — the same IR works for both. The default is `CPU` so the OpenVINO
  detector keeps the iGPU.
- **Changing the model** needs no code change: re-run the prep script with a
  different `--repo`, restart the service, and record the new row here.
