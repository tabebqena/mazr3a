"""FastAPI app for the portal: auth + proxied Frigate JSON/media + Firewatch
evidence, plus the static SPA. See portal/__init__.py and plans/portal-web-app.md.
"""
import asyncio
import contextlib
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from typing import Optional

import httpx
import websockets
from fastapi import (
    Depends, FastAPI, HTTPException, Request, Response, WebSocket,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse,
)

from portal import auth, config as pconf, eventstore, firestore, frigate

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONF_PATH = os.environ.get("PORTAL_CONF", "/config/portal.conf")
COOKIE_NAME = "portal_session"
# App/UI version - the SINGLE source of truth, shown as the bottom-most version
# label in the SPA footer. It is (a) injected into the served index.html at the
# {{ APP_VERSION }} token (see _render_index below, so the pre-JS fallback label
# never drifts from this constant) and (b) served to the client via
# /api/settings -> app_version (app.js re-fills #ver-no after boot). Bump on
# UI/API change; no other copy of the number should be kept in the front-end.
# AI instruction : update the version with each task, so the user can see the
# changes in the footer of the SPA. The version is also used in the API
# responses to indicate the version of the portal. APP_VERSION is ALSO an input
# to the static-asset fingerprint below, so bumping it (on every update)
# rotates the fingerprinted /static/* filenames and forces browsers to load the
# fresh app.js/style.css instead of a stale cached copy.
APP_VERSION = "0.4.0"
_HTMX = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg, users = pconf.parse_file(CONF_PATH)
    app.state.cfg = cfg
    app.state.users = users
    app.state.secret = pconf.get(cfg, "SECRET_KEY")
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        follow_redirects=True,
    )
    if not app.state.secret:
        print("WARNING: portal.conf SECRET_KEY is empty - sessions will not work")
    if not users:
        print("WARNING: no users configured in portal.conf - login is impossible")
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="mazr3a CCTV portal",
    description=(
        "Authenticated API for the Frigate + Firewatch stack. Proxies Frigate "
        "events/media, serves Firewatch fire evidence, and serves the Live view "
        "as same-origin go2rtc MSE over WebSocket (via /api/live/<cam>/mse), "
        "with a same-origin HLS fallback (via /api/live/<cam>/hls)."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# --------------------------------------------------------------------------
# helpers / dependencies
# --------------------------------------------------------------------------
def _cfg(request: Request):
    return request.app.state.cfg


def _ttl(request: Request):
    days = pconf.geti(_cfg(request), "SESSION_DAYS", 7)
    return days * 86400


def _set_session(response: Response, username: str, request: Request):
    token = auth.make_session(request.app.state.secret, username, _ttl(request))
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=_ttl(request), httponly=True,
        samesite="lax", secure=pconf.getb(_cfg(request), "COOKIE_SECURE", True),
        path="/",
    )


def _clear_session(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")


def current_user(request: Request):
    """Dependency: the authenticated user dict or a 401."""
    if not request.app.state.secret:
        raise HTTPException(status_code=401, detail="not configured")
    token = request.cookies.get(COOKIE_NAME)
    name = auth.read_session(request.app.state.secret, token)
    if not name:
        raise HTTPException(status_code=401, detail="not authenticated")
    for user in request.app.state.users:
        if user["username"] == name:
            return user
    raise HTTPException(status_code=401, detail="not authenticated")


def current_admin(request: Request, user: dict = Depends(current_user)):
    """Dependency: like current_user but ONLY the admin user (403 otherwise).

    The admin is the user whose username is `admin` (see plans/
    portal-admin-debug-logs.md). Front-end nav gating uses /api/me is_admin,
    but this server-side check is the real gate for admin-only routes.
    """
    if (user.get("username") or "").lower() != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    return user


def _frigate_base(request: Request):
    return pconf.get(_cfg(request), "FRIGATE_API", "http://frigate:5000")


# --------------------------------------------------------------------------
# app / static - content-hash fingerprinting (cache busting)
# --------------------------------------------------------------------------
# index.html carries a {{ APP_VERSION }} token as the fallback text of the
# bottom-most version label, plus {{ ASSET_* }} tokens for every cacheable
# static asset. _render_index() substitutes them on every request, so the
# labels/URLs always match the running build without hard-coded copies.
#
# CACHE BUSTING: browsers cache /static/* by URL, so an updated app.js behind
# the SAME filename would be served from cache (stale UI). Every cacheable
# asset is therefore served under a CONTENT-HASH fingerprinted filename, e.g.
# /static/app-154kuhn7.js, derived from APP_VERSION + the file's bytes. Bumping
# APP_VERSION OR editing the file changes the fingerprint, so the URL changes
# and the browser is forced to fetch the new version. index.html is served
# no-cache (below) and always references the CURRENT fingerprinted names.
#
# AI instruction: on every portal update bump APP_VERSION above - the
# fingerprinting below is automatic. NEVER hard-code /static/... filenames in
# index.html; always use the {{ ASSET_* }} tokens so _render_index substitutes
# the current fingerprint. See .roo/rules/portal-cache-busting.md.

# Cacheable assets to fingerprint: (relpath under STATIC_DIR, {{ TOKEN }}).
CACHEABLE_ASSETS = [
    ("style.css", "ASSET_STYLE"),
    ("app.js", "ASSET_APP"),
    ("vendor/hls.min.js", "ASSET_HLS"),
    ("favicon.svg", "ASSET_FAVICON"),
]
# Fingerprinted public name -> real relpath, e.g. "app-154kuhn7.js" -> "app.js".
ASSET_ALIAS: dict = {}
# {{ TOKEN }} -> fingerprinted public name used by _render_index().
_ASSET_FINGERPRINTS: dict = {}


def _asset_suffix(relpath: str) -> str:
    """Short base36 fingerprint of APP_VERSION + file bytes (like '154kuhn7')."""
    data = (STATIC_DIR / relpath).read_bytes()
    digest = hashlib.sha256()
    digest.update(APP_VERSION.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(data)
    n = int(digest.hexdigest()[:16], 16)  # first 64 bits of the SHA-256
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    for _ in range(8):
        n, rem = divmod(n, 36)
        out.append(chars[rem])
    return "".join(reversed(out))


def _build_asset_manifest():
    for relpath, token in CACHEABLE_ASSETS:
        base = Path(relpath).name              # e.g. hls.min.js / app.js
        stem, dot, ext = base.rpartition(".")
        name = base if not dot else f"{stem}-{_asset_suffix(relpath)}{dot}{ext}"
        parent = Path(relpath).parent
        public = str(parent / name) if str(parent) != "." else name
        _ASSET_FINGERPRINTS[token] = public
        ASSET_ALIAS[public] = relpath


_build_asset_manifest()


def _render_index() -> str:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{ APP_VERSION }}", APP_VERSION)
    for token, public in _ASSET_FINGERPRINTS.items():
        html = html.replace("{{ " + token + " }}", f"/static/{public}")
    return html


@app.get("/", include_in_schema=False)
async def index():
    # no-cache: the page carries the live APP_VERSION + current asset URLs -
    # always revalidate so a bumped version/fresh fingerprint is picked up
    # immediately instead of a stale cached copy.
    return HTMLResponse(
        content=_render_index(),
        headers={"Cache-Control": "no-cache"},
    )


# Replaces the plain StaticFiles mount: serves each /static asset and applies
# cache headers. Fingerprinted names (in ASSET_ALIAS) are immutable - safe to
# cache for a year because the URL changes whenever the content or APP_VERSION
# changes. Any other file under /static (legacy/hard-coded references) is
# served current-but-no-cache so a stale URL still gets fresh content.
@app.get("/static/{path:path}", include_in_schema=False)
async def static_asset(path: str):
    real = ASSET_ALIAS.get(path)
    if real is not None:
        return FileResponse(
            STATIC_DIR / real,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    candidate = (STATIC_DIR / path).resolve()
    if not str(candidate).startswith(str(STATIC_DIR)) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="static asset not found")
    return FileResponse(candidate, headers={"Cache-Control": "no-cache"})


# Browsers that ignore <link rel="icon"> and request /favicon.ico by default
# get the same SVG asset instead of a 404.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------
@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    username = str(body.get("username") or "").strip()
    password = str(body.get("password") or "")
    for user in request.app.state.users:
        if user["username"] == username and auth.verify_password(
                password, user["password_hash"]):
            response = JSONResponse({"ok": True, "username": username,
                                     "default_camera": user["default_camera"]})
            _set_session(response, username, request)
            return response
    raise HTTPException(status_code=401, detail="invalid credentials")


@app.post("/api/logout")
async def logout():
    response = JSONResponse({"ok": True})
    _clear_session(response)
    return response


@app.get("/api/me")
async def me(request: Request, user: dict = Depends(current_user)):
    return {
        "username": user["username"],
        "is_admin": (user.get("username") or "").lower() == "admin",
        "default_camera": (user.get("default_camera")
                           or pconf.get(_cfg(request), "DEFAULT_CAMERA")),
    }


# --------------------------------------------------------------------------
# settings + cameras
# --------------------------------------------------------------------------
@app.get("/api/settings")
async def settings(request: Request, user: dict = Depends(current_user)):
    cfg = _cfg(request)
    default_cam = user.get("default_camera") or pconf.get(cfg, "DEFAULT_CAMERA")
    return {
        "username": user["username"],
        "default_camera": default_cam,
        "stream_idle_timeout_s": pconf.geti(cfg, "STREAM_IDLE_TIMEOUT_S", 30),
        "app_version": APP_VERSION,
    }


@app.get("/api/cameras")
async def cameras(request: Request, user: dict = Depends(current_user)):
    client = request.app.state.client
    try:
        items = await frigate.camera_list(client, _frigate_base(request))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"frigate unreachable: {exc}")
    cfg = _cfg(request)
    default_cam = user.get("default_camera") or pconf.get(cfg, "DEFAULT_CAMERA")
    return {"default_camera": default_cam, "cameras": items}


async def _frigate_stream(request: Request, path: str, params=None):
    """Stream an upstream Frigate route verbatim (content-type preserved).

    go2rtc is embedded in Frigate and its HTTP API is reverse-proxied by
    Frigate under /api/go2rtc/* on port 5000. Requests here carry the session
    cookie (same-origin) and are relayed server-side, so the go2rtc HLS tree
    never leaves the portal origin / Cloudflare Access boundary.
    """
    client = request.app.state.client
    req = client.build_request("GET", _frigate_base(request) + path,
                               params=params)
    try:
        resp = await client.send(req, stream=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"frigate unreachable: {exc}")
    if resp.status_code != 200:
        await resp.aclose()
        raise HTTPException(status_code=resp.status_code, detail="frigate error")

    async def gen():
        try:
            async for chunk in resp.aiter_bytes(64 * 1024):
                yield chunk
        finally:
            await resp.aclose()

    ctype = resp.headers.get("content-type") or "application/octet-stream"
    return StreamingResponse(gen(), media_type=ctype,
                             headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------
# Live view: same-origin go2rtc transport, proxied through the portal.
#
# PRIMARY - MSE over WebSocket (/api/live/<cam>/mse): the SPA opens a WS here
# and this route relays it to Frigate's embedded go2rtc
# ws://<frigate>/api/go2rtc/api/ws?src=<cam>_portal. Frigate only nginx-proxies
# go2rtc's own /api/ws (go2rtc serves it to any client); the portal keeps the
# whole exchange SAME-ORIGIN behind the session cookie + CF Access boundary.
# This is the same transport Frigate's own UI uses and it plays quiet cameras
# (sparse keyframes) reliably - unlike go2rtc HLS, whose ~1 s sliding window
# with pinned EXT-X-MEDIA-SEQUENCE makes hls.js stall/skip on sparse-keyframe
# cameras (see plans/investigate-hls-loop-quiet-cams.md).
#
# FALLBACK - HLS (/api/live/<cam>/hls/*) for browsers with no MediaSource
# (e.g. old iOS native HLS): the go2rtc master playlist + media/segments are
# proxied verbatim:
#   GET /api/live/<cam>/hls/stream.m3u8          -> go2rtc master playlist
#   GET /api/live/<cam>/hls/hls/<playlist|seg>.ts -> go2rtc media/segments
#
# AUDIO: the cameras send G.711 (PCMU), which browsers cannot decode, so both
# transports use the <cam>_portal go2rtc source (config/config.yaml
# go2rtc.streams) - an on-demand ffmpeg transcode that copies the video and
# re-encodes the audio to AAC. The SPA's Live stage exposes a muted-by-default
# 🔊 toggle (browsers only allow sound autoplay after a user gesture).
# --------------------------------------------------------------------------
# go2rtc source suffix for the AAC-audio portal variants (config/config.yaml
# go2rtc.streams). Keep these two files in sync.
LIVE_SOURCE_SUFFIX = "_portal"

@app.get("/api/live/{camera}/hls/stream.m3u8")
async def live_hls_master(camera: str, request: Request,
                          user: dict = Depends(current_user)):
    """go2rtc HLS master playlist for a camera (same-origin, behind the cookie).

    Serves the <cam>_portal AAC-audio variant so browsers can decode the
    audio (the base <cam> stream carries G.711/PCMU, not MSE-playable).
    """
    if not frigate.valid_camera_name(camera):
        raise HTTPException(status_code=400, detail="invalid camera name")
    return await _frigate_stream(
        request, "/api/go2rtc/api/stream.m3u8",
        params={"src": camera + LIVE_SOURCE_SUFFIX})


@app.get("/api/live/{camera}/hls/{rest:path}")
async def live_hls_media(camera: str, rest: str, request: Request,
                         user: dict = Depends(current_user)):
    """go2rtc HLS media playlist + TS segments for a camera.

    The master references 'hls/playlist.m3u8' (relative), which a browser
    resolves to /api/live/<cam>/hls/hls/playlist.m3u8; segments resolve
    similarly under hls/. Both are forwarded verbatim to go2rtc.
    """
    if not frigate.valid_camera_name(camera):
        raise HTTPException(status_code=400, detail="invalid camera name")
    if not rest.startswith("hls/"):
        raise HTTPException(status_code=404, detail="not found")
    return await _frigate_stream(request, "/api/go2rtc/api/" + rest,
                                 params=request.query_params)


@app.websocket("/api/live/{camera}/mse")
async def live_mse(camera: str, websocket: WebSocket):
    """Same-origin go2rtc MSE-over-WebSocket tunnel (primary Live transport).

    The SPA opens ws://<portal>/api/live/<cam>/mse and this relays text/binary
    frames to Frigate's embedded go2rtc (ws://<frigate>/api/go2rtc/api/ws?src=
    <cam>_portal). The browser speaks go2rtc's MSE protocol (send {"type":"mse"},
    then receive a codec text frame followed by fMP4 init + media segments); the
    relay is transport-agnostic and preserves the message type. Same-origin keeps
    it behind the portal session cookie + CF Access; the <cam>_portal source
    carries AAC audio so Live keeps sound.

    Auth: the session cookie is validated here (FastAPI's HTTP dependency cannot
    be used on a WebSocket); invalid/unknown users are closed with code 4401.
    """
    await websocket.accept()
    secret = websocket.app.state.secret
    token = websocket.cookies.get(COOKIE_NAME)
    name = auth.read_session(secret, token) if secret else None
    authed = bool(name) and any(u["username"] == name
                                for u in websocket.app.state.users)
    if not authed or not frigate.valid_camera_name(camera):
        await websocket.close(code=4401)
        return

    base = pconf.get(websocket.app.state.cfg, "FRIGATE_API", "http://frigate:5000")
    if base.startswith("https://"):
        ws_base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        ws_base = "ws://" + base[len("http://"):]
    else:
        ws_base = "ws://" + base
    upstream_url = (ws_base.rstrip("/") + "/api/go2rtc/api/ws?src="
                    + quote(camera + LIVE_SOURCE_SUFFIX, safe=""))
    try:
        upstream = await websockets.connect(
            upstream_url, max_size=None, ping_interval=20, ping_timeout=20,
            open_timeout=10)
    except Exception:
        await websocket.close(code=1011)
        return

    async def client_to_upstream():
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    await upstream.send(message["text"])
                elif message.get("bytes") is not None:
                    await upstream.send(message["bytes"])
        except Exception:
            pass

    async def upstream_to_client():
        try:
            async for message in upstream:
                if isinstance(message, (bytes, bytearray)):
                    await websocket.send_bytes(bytes(message))
                else:
                    await websocket.send_text(message)
        except Exception:
            pass

    tasks = [asyncio.create_task(client_to_upstream()),
             asyncio.create_task(upstream_to_client())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        with contextlib.suppress(Exception):
            await upstream.close()
        with contextlib.suppress(Exception):
            await websocket.close()


@app.get("/api/live/{camera}/latest.jpg")
async def live_snapshot(camera: str, request: Request,
                        user: dict = Depends(current_user)):
    """Live snapshot fallback - Frigate's detect frame (~1 fps, no extra decode).

    Used when MSE is unavailable (MediaSource unsupported or the stream URL not
    configured). Frigate serves /api/<camera>/latest.jpg from its already-
    decoded detect frame, so this proxy JPEG adds ~no host load.
    """
    if not frigate.valid_camera_name(camera):
        raise HTTPException(status_code=400, detail="invalid camera name")
    return await _frigate_media(request, "/api/{}/latest.jpg".format(camera),
                                "image/jpeg")


# --------------------------------------------------------------------------
# Frigate JSON proxy (events)
# --------------------------------------------------------------------------
@app.get("/api/events")
async def events(request: Request, user: dict = Depends(current_user)):
    client = request.app.state.client
    try:
        resp = await client.get(
            _frigate_base(request) + "/api/events",
            params=request.query_params,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"frigate unreachable: {exc}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code,
                            detail="frigate error")
    return JSONResponse(content=resp.json())


# --------------------------------------------------------------------------
# Frigate media proxy (event snapshot / clip) - streamed, gated by the portal
# cookie (the SPA is same-origin so the cookie flows on <img>/fetch too).
# --------------------------------------------------------------------------
async def _frigate_media(request: Request, path: str, media_type: str):
    client = request.app.state.client
    req = client.build_request("GET", _frigate_base(request) + path,
                               params=request.query_params)
    try:
        resp = await client.send(req, stream=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"frigate unreachable: {exc}")
    if resp.status_code != 200:
        await resp.aclose()
        raise HTTPException(status_code=resp.status_code, detail="frigate error")

    async def gen():
        try:
            async for chunk in resp.aiter_bytes(64 * 1024):
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(gen(), media_type=media_type,
                             headers={"Cache-Control": "no-store"})


@app.get("/api/events/{event_id}/snapshot.jpg")
async def event_snapshot(event_id: str, request: Request,
                          user: dict = Depends(current_user)):
    return await _frigate_media(
        request, f"/api/events/{event_id}/snapshot.jpg", "image/jpeg")


@app.get("/api/events/{event_id}/clip.mp4")
async def event_clip(event_id: str, request: Request,
                     user: dict = Depends(current_user)):
    return await _frigate_media(
        request, f"/api/events/{event_id}/clip.mp4", "video/mp4")


# --------------------------------------------------------------------------
# Firewatch evidence
# --------------------------------------------------------------------------
@app.get("/api/fire/events")
async def fire_events(request: Request, user: dict = Depends(current_user),
                      camera: Optional[str] = None, alerted: Optional[int] = None,
                      label: Optional[str] = None,
                      after: Optional[float] = None, before: Optional[float] = None,
                      limit: int = 50, offset: int = 0):
    cfg = _cfg(request)
    db = pconf.get(cfg, "FIREWATCH_DB", "/media/firewatch.db")
    result = firestore.list_frames(
        db,
        camera=camera or None,
        alerted=None if alerted is None else bool(alerted),
        label=label or None,
        after=after, before=before,
        limit=limit, offset=offset,
    )
    return result


@app.get("/api/fire/{frame_id}/image.jpg")
async def fire_image(frame_id: int, request: Request,
                     user: dict = Depends(current_user)):
    cfg = _cfg(request)
    db = pconf.get(cfg, "FIREWATCH_DB", "/media/firewatch.db")
    path = firestore.frame_image_path(cfg, db, frame_id)
    if not path:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=300"})


# --------------------------------------------------------------------------
# scenereader - cross-camera EPISODES + the per-capture scene log
#
# scenereader (scenereader/scenereader.py) is the single writer of a WAL SQLite
# DB in its own store dir (default ./media/events/ on the host, seen here as
# /media/events). The portal only ever READS it (eventstore.py uses PRAGMA
# query_only).
#
# NO IMAGE PATHS ARE TOUCHED HERE: every visit/event carries the Frigate event
# id, and the frame is served by the EXISTING Frigate proxy defined above
# (/api/events/<id>/snapshot.jpg). scenereader never copies a frame and the
# portal never reads a filesystem path out of a DB row.
#
# /api/scenes is KEPT as a thin COMPATIBILITY SHIM over the same data so the
# existing Scenes tab keeps working until the SPA grows the Episodes view; it
# returns the OLD field names and maps `reason` onto the event label.
# --------------------------------------------------------------------------
def _scenereader_paths(request: Request):
    """(db, status_file, trigger_file) resolved from portal.conf.

    Defaults match the compose service's STORE_DIR so no portal.conf change is
    required, but SCENEREADER_DB / SCENEREADER_STATUS / SCENEREADER_TRIGGER can
    override each.
    """
    cfg = _cfg(request)
    db = pconf.get(cfg, "SCENEREADER_DB", "/media/events/events.db")
    store = eventstore.store_dir_from_db(db)
    status = pconf.get(cfg, "SCENEREADER_STATUS", "") or os.path.join(
        store, eventstore.STATUS_NAME)
    trigger = pconf.get(cfg, "SCENEREADER_TRIGGER", "") or os.path.join(
        store, eventstore.TRIGGER_NAME)
    return db, status, trigger


@app.get("/api/episodes")
async def episodes(request: Request, user: dict = Depends(current_user),
                   day: Optional[str] = None, camera: Optional[str] = None,
                   after: Optional[float] = None, before: Optional[float] = None,
                   with_visits: Optional[int] = None,
                   limit: int = 50, offset: int = 0):
    """The cross-camera person stories (the required output).

    Each item carries BOTH narratives - `narrative` (English) and
    `narrative_ar` (Arabic) - built deterministically by scenereader from the
    same facts, plus the visits with a Frigate snapshot URL per capture.
    """
    db, _status, _trigger = _scenereader_paths(request)
    result = eventstore.list_episodes(
        db, day=day or None, camera=camera or None,
        after=after, before=before,
        with_visits=bool(with_visits), limit=limit, offset=offset)
    result["days"] = eventstore.days(db)
    return result


@app.get("/api/episodes/{episode_id}")
async def episode_detail(episode_id: int, request: Request,
                         user: dict = Depends(current_user)):
    db, _status, _trigger = _scenereader_paths(request)
    item = eventstore.get_episode(db, episode_id)
    if item is None:
        raise HTTPException(status_code=404, detail="episode not found")
    return item


@app.get("/api/scenelog")
async def scenelog(request: Request, user: dict = Depends(current_user),
                   camera: Optional[str] = None, label: Optional[str] = None,
                   min_tier: Optional[str] = None,
                   has_caption: Optional[int] = None,
                   after: Optional[float] = None, before: Optional[float] = None,
                   sort: str = "time", limit: int = 50, offset: int = 0):
    """The per-capture scene log: deterministic description (+ optional VLM
    caption) and its importance tier. `min_tier` = "at least this important"."""
    db, _status, _trigger = _scenereader_paths(request)
    return eventstore.list_events(
        db, camera=camera or None, label=label or None,
        min_tier=min_tier or None,
        has_caption=None if has_caption is None else bool(has_caption),
        after=after, before=before,
        sort=sort if sort in ("importance", "time") else "time",
        limit=limit, offset=offset)


@app.get("/api/scenereader/status")
async def scenereader_status(request: Request, user: dict = Depends(current_user)):
    """The service heartbeat plus whether its store is reachable."""
    db, status_file, _trigger = _scenereader_paths(request)
    status = eventstore.read_status(status_file)
    status["store_present"] = os.path.exists(db)
    if not status.get("updated_at"):
        status["note"] = "scenereader has not written a status file yet"
    return status


@app.post("/api/scenereader/drain")
async def scenereader_drain(request: Request, user: dict = Depends(current_user)):
    """Ask scenereader for a caption batch now (the "Process now" button).

    Writes the trigger flag file it polls; the service still honours its idle
    governor, so this can never force a hot/overloaded host to work.
    """
    _db, _status, trigger = _scenereader_paths(request)
    if not eventstore.request_drain(trigger):
        raise HTTPException(status_code=503,
                            detail="could not write the trigger file")
    return {"requested": True}


# --------------------------------------------------------------------------
# Scenes tab COMPATIBILITY SHIM (scenewatch -> scenereader)
#
# The SPA's Scenes tab still calls /api/scenes and /api/scenes/<id>/image.jpg.
# Rather than break it while the Episodes view is built, these two endpoints
# keep the OLD response shape over the NEW data:
#   * `reason` (motion|baseline) maps onto the event LABEL filter
#   * `description` = the VLM caption when present, else the deterministic one
#   * the image is a REDIRECT to the existing Frigate snapshot proxy, because
#     scenereader keeps no image paths of its own.
# Remove both once the SPA uses only /api/scenelog + /api/episodes.
# --------------------------------------------------------------------------
@app.get("/api/scenes")
async def scenes_compat(request: Request, user: dict = Depends(current_user),
                        camera: Optional[str] = None, reason: Optional[str] = None,
                        min_tier: Optional[str] = None,
                        with_image: Optional[int] = None,
                        after: Optional[float] = None, before: Optional[float] = None,
                        sort: str = "importance", limit: int = 50, offset: int = 0):
    db, _status, _trigger = _scenereader_paths(request)
    res = eventstore.list_events(
        db, camera=camera or None, label=reason or None,
        min_tier=min_tier or None, after=after, before=before,
        sort=sort if sort in ("importance", "time") else "importance",
        limit=limit, offset=offset)
    items = []
    for event in res.get("items", []):
        if with_image is not None and bool(event.get("image_url")) != bool(with_image):
            continue
        items.append({
            "id": event["id"],
            "camera": event["camera"],
            "captured_at": event["start_time"],
            "ts_utc": "",
            "description": event.get("description_vlm")
            or event.get("description_meta") or "",
            "reason": event.get("label") or "",
            "model": "scenereader",
            "latency_ms": 0,
            "importance": event.get("importance", 0),
            "tier": event.get("tier", "normal"),
            "has_image": bool(event.get("image_url")),
            "image_url": ("/api/scenes/{}/image.jpg".format(event["id"])
                          if event.get("image_url") else None),
        })
    return {"items": items, "total": res.get("total", len(items)),
            "note": res.get("note")}


@app.get("/api/scenes/{scene_id}/image.jpg")
async def scene_image_compat(scene_id: int, request: Request,
                             user: dict = Depends(current_user)):
    """Redirect to Frigate's own snapshot proxy for this capture."""
    db, _status, _trigger = _scenereader_paths(request)
    item = eventstore.get_event(db, scene_id)
    if not item or not item.get("image_url"):
        raise HTTPException(status_code=404, detail="image not found")
    return RedirectResponse(url=item["image_url"], status_code=307)


# --------------------------------------------------------------------------
# Admin-only Debug tab: last log messages of containers.
#
# The portal container itself has no Docker access; it pulls on demand from the
# read-only `logs` sidecar (scripts/container_logs.py, compose service `logs`)
# over the internal compose network. The sidecar owns the host Docker socket
# and stays idle - it is queried only when the admin opens the Debug tab.
#
# Two admin-only endpoints: /api/admin/containers (lightweight list, no logs,
# used to populate the UI's container dropdown) and /api/admin/logs, which
# returns log tails either for ONE selected container (?name=<c>) or for every
# container when no name is given (the Debug tab's "All containers" option).
# --------------------------------------------------------------------------
def _logs_base(request: Request):
    return pconf.get(_cfg(request), "LOGS_API", "http://logs:8090").rstrip("/")


async def _sidecar_containers(client, base):
    """Fetch the sidecar's container list; raise 503/502 on transport/HTTP errors."""
    try:
        resp = await client.get(base + "/api/containers", timeout=8.0)
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"logs sidecar unreachable: {exc}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"logs sidecar error: HTTP {resp.status_code}")
    try:
        return (resp.json() or {}).get("containers") or []
    except Exception:
        return []


@app.get("/api/admin/containers")
async def admin_containers(request: Request,
                           admin: dict = Depends(current_admin)):
    """Debug tab: lightweight list of every container (admin only, no logs).

    Feeds the front-end's "Container" dropdown. 503 = sidecar unreachable;
    502 = sidecar/docker error.
    """
    client = request.app.state.client
    base = _logs_base(request)
    containers = await _sidecar_containers(client, base)
    return {"containers": containers}


@app.get("/api/admin/logs")
async def admin_logs(request: Request, tail: int = 200,
                     name: Optional[str] = None,
                     admin: dict = Depends(current_admin)):
    """Debug tab: last `tail` log lines of one container or every container.

    `name` selects a single container (the UI's per-container view); without
    it every container is returned (the "All containers" view). Each container
    carries its merged stdout+stderr tail (timestamps). 503 = logs sidecar
    unreachable; 502 = sidecar/docker error; a per-container `error` field
    records failures fetching that one container's logs.
    """
    client = request.app.state.client
    base = _logs_base(request)
    tail = max(1, min(2000, tail))
    containers = await _sidecar_containers(client, base)
    if name:
        # Focus on the selected container only. If it vanished between the list
        # and the logs fetch, keep a stub entry so the UI can show an error
        # instead of silently rendering nothing.
        needle = name.strip().lower()
        containers = [c for c in containers
                      if (c.get("name") or "").lower() == needle]
        if not containers:
            containers = [{"name": name.strip(), "state": "?",
                           "status": "", "image": ""}]
    result = []
    for c in containers:
        entry = {
            "name": c.get("name"),
            "state": c.get("state"),
            "status": c.get("status"),
            "image": c.get("image"),
            "logs": "",
            "error": None,
        }
        try:
            r = await client.get(base + "/api/logs",
                                 params={"name": c.get("name"), "tail": tail},
                                 timeout=8.0)
            if r.status_code == 200:
                entry["logs"] = (r.json() or {}).get("logs") or ""
            else:
                entry["error"] = f"HTTP {r.status_code}"
        except Exception as exc:
            entry["error"] = str(exc)
        result.append(entry)
    return {"tail": tail, "name": (name.strip() if name else None),
            "containers": result}
