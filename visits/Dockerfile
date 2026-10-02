# ============================================================
# visits - per-camera person-visit log runtime image.
#
# Dependency-only image: NO application code is baked in. At runtime compose
# mounts read-only:
#   ./visits -> /visits   (visits.py)
#   ./config -> /config   (visits.conf, places.conf)
# and read-write:
#   ./media  -> /media    (Frigate frames read in place + the text store at
#                          /media/visits/)
# so editing code or config needs only `docker compose restart visits`, never an
# image rebuild.
#
# Far smaller than the retired scenereader image: there is no VLM and no
# OpenVINO/llama.cpp - just Pillow + numpy for the clothing histogram.
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Keep versions pinned/lower-bounded so a future rebuild stays reproducible.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user: the container reads mounted frames, writes ONLY its own
# text store under /media/visits, and calls Frigate's REST API.
RUN useradd --create-home --uid 1000 visits
USER visits

# visits.py reads VISITS_CONF (default /config/visits.conf).
ENTRYPOINT ["python", "/visits/visits.py"]
