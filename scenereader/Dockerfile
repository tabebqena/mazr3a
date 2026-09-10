# ============================================================
# scenereader - cross-camera episode narrator runtime image.
#
# Dependency-only image: NO application code is baked in. At runtime compose
# mounts:
#   ./scenereader -> /scenereader   (the watcher: scenereader.py + its modules)
#   ./config      -> /config        (scenereader.conf, places.conf,
#                                    frigate.db for the read-only metadata join)
#   ./models      -> /models        (the small GGUF, the optional llama.cpp
#                                    binaries in models/scene/bin/, and the
#                                    RETAINED 2B OpenVINO IR in models/scene/)
# and read-write:
#   ./media       -> /media         (Frigate's frames read in place + the text
#                                    store at /media/events/)
# so editing code or config needs only `docker compose restart scenereader`,
# never an image rebuild.
#
# WHY THIS IMAGE IS BIGGER THAN firewatch's: it carries the OpenVINO GenAI
# runtime so the RETAINED 2B model keeps working without a rebuild. The default
# backend (llama.cpp) uses a host-provided binary, not a pip package.
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Keep versions pinned/lower-bounded so a future rebuild stays reproducible.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user: the container reads mounted files, writes ONLY its own text
# store under /media/events, and makes outbound calls to Frigate.
RUN useradd --create-home --uid 1000 scenereader
USER scenereader

# scenereader.py reads SCENEREADER_CONF (default /config/scenereader.conf) and
# PLACES_CONF (default /config/places.conf).
ENTRYPOINT ["python", "/scenereader/scenereader.py"]
