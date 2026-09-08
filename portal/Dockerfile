# ============================================================
# portal - dependency-only runtime image for the Frigate+Firewatch
# web portal (FastAPI backend + vanilla JS SPA). Like firewatch, the
# image carries ONLY Python + deps; application code, config and the
# firewatch evidence media are mounted at runtime (read-only), so code
# edits need only `docker compose restart portal` - no image rebuild.
#
# Mounts (see docker-compose.yml):
#   ./portal -> /portal:ro   (portal package + static SPA)
#   ./config -> /config:ro   (portal.conf: users/secret/tunables)
#   ./media  -> /media       (firewatch evidence; rw because SQLite WAL
#                             readers need -shm/-wal write access)
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/portal \
    PORTAL_CONF=/config/portal.conf

WORKDIR /portal

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && useradd --uid 1000 --create-home portal \
 && mkdir -p /portal /config /media \
 && chown portal:portal /tmp

USER portal

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "portal.app:app", \
     "--host", "0.0.0.0", "--port", "8080"]
