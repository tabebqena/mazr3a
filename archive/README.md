# archive/

Retired-but-kept pieces. Nothing here is part of a normal
`docker compose up -d --build` (see `dev_scripts/deploy_all.sh`) — it exists so a
past design can be restored deliberately, never by accident.

## `scenereader.compose.yml`

The **compose definition** of the retired `scenereader` service
(cross-camera episode narrator + VLM captioner). On 2026-10-02 it was removed
from the root `docker-compose.yml` and the `visits` service took its place, so a
normal deploy can no longer **build or start** it.

What was kept and what was not:

| Path | State |
|---|---|
| `scenereader/` (code, Dockerfile, requirements) | **kept** — dormant, not built |
| `config/scenereader.conf`, `config/places.conf` | **kept** (places.conf is also used by `visits`) |
| `models/scene/` | **kept** — git-ignored, never deleted/re-downloaded |
| `media/events/` (its store) | **left in place** — no rows were moved or deleted |
| its compose service block | **moved here** |

### Restoring it

Do **not** just copy this file to the repo root. Its volume paths are relative
to the repo root, and it needs to be merged back under the root file's
`services:` key:

1. Copy the `scenereader:` block from `scenereader.compose.yml` into
   `docker-compose.yml` under `services:`.
2. `docker compose up -d --build scenereader`.
3. Undo the `visits` service if it is to be replaced.

### Removing the stopped container

`docker compose up` does **not** remove a container whose service was deleted
from the compose file. The old `scenereader` container is simply left stopped.
To delete it (and only it) explicitly:

```bash
docker rm -f scenereader    # only if you are sure
```
