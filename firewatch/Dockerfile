# ============================================================
# firewatch - farm fire/smoke watchdog runtime image.
#
# Dependency-only image: NO application code is baked in. At runtime the
# compose service mounts read-only:
#   ./scripts -> /scripts   (firewatch.py + monitor_lib.py)
#   ./config  -> /config    (firewatch.conf, telegram.conf)
#   ./models  -> /models    (fire/smoke OpenVINO IR model)
# so editing code or config needs only `docker compose restart firewatch`,
# never an image rebuild.
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Keep versions pinned/lower-bounded so a future rebuild stays reproducible.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user: the container only reads mounted files and makes
# outbound HTTP calls to Frigate (frigate:5000) and the Telegram Bot API.
RUN useradd --create-home --uid 1000 firewatch
USER firewatch

# firewatch.py reads FIREWATCH_CONF (default /config/firewatch.conf) and the
# Telegram creds from /config/telegram.conf (mounted read-only, git-ignored).
ENTRYPOINT ["python", "/scripts/firewatch.py"]
