# ============================================================
# scenewatch - farm scene-description watchdog runtime image.
#
# Dependency-only image: NO application code is baked in. At runtime the
# compose service mounts read-only:
#   ./scenewatch -> /scenewatch  (scenewatch.py, the watcher itself)
#   ./config     -> /config      (scenewatch.conf)
#   ./models     -> /models      (the SmolVLM OpenVINO IR at /models/scene)
# and read-write:
#   ./media      -> /media       (the scene DB at /media/scenewatch/)
# so editing code or config needs only `docker compose restart scenewatch`,
# never an image rebuild.
#
# This image is heavier than firewatch's: the OpenVINO GenAI runtime adds a
# few hundred MB. It is built once and cached by compose.
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Keep versions pinned/lower-bounded so a future rebuild stays reproducible.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user: the container only reads mounted files, makes outbound
# HTTP calls to Frigate (frigate:5000) and writes its own scene store.
RUN useradd --create-home --uid 1000 scenewatch
USER scenewatch

# scenewatch.py reads SCENEWATCH_CONF (default /config/scenewatch.conf) and
# loads the VLM from MODEL_DIR (default /models/scene).
ENTRYPOINT ["python", "/scenewatch/scenewatch.py"]
