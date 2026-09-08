# ============================================================
# portal - dependency-only runtime image for the Frigate+Firewatch
# web portal (FastAPI backend + vanilla JS SPA). Like firewatch, the
# image carries ONLY Python + deps; application code, config and the
# firewatch evidence media are mounted at runtime (read-only), so code
# edits need only `docker compose restart portal` - no image rebuild.
#
# Mounts (see docker-compose.yml):
#   ./portal -> /srv/app/portal:ro   (portal package + static SPA)
#   ./config -> /config:ro           (portal.conf: users/secret/tunables)
#   ./media  -> /media               (firewatch evidence; rw because SQLite WAL
#                                     readers need -shm/-wal write access)
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv/app \
    PORTAL_CONF=/config/portal.conf

# App code is bind-mounted at runtime at /srv/app/portal, so the `portal`
# package is importable from /srv/app (portal.app).
WORKDIR /srv/app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && useradd --uid 1000 --create-home portal \
 && mkdir -p /srv/app /config /media \
 && chown portal:portal /tmp

USER portal

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "portal.app:app", \
     "--host", "0.0.0.0", "--port", "8080"]
