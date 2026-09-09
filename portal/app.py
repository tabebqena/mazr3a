"""FastAPI app for the portal: auth + proxied Frigate JSON/media + Firewatch
evidence, plus the static SPA. See portal/__init__.py and plans/portal-web-app.md.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Optional

import httpx
from fastapi import (
    Depends, FastAPI, HTTPException, Request, Response,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from portal import auth, config as pconf, firestore, frigate

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONF_PATH = os.environ.get("PORTAL_CONF", "/config/portal.conf")
COOKIE_NAME = "portal_session"
# App/UI version - the SINGLE source of truth, shown as the bottom-most version
# label in the SPA footer. It is (a) injected into the served index.html at the
# {{ APP_VERSION }} token (see _render_index below, so the pre-JS fallback label
# never drifts from this constant) and (b) served to the client via
# /api/settings -> app_version (app.js re-fills #ver-no after boot). Bump on
# UI/API change; no other copy of the number should be kept in the front-end.
APP_VERSION = "0.3.4"
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
        "as same-origin HLS from Frigate's embedded go2rtc (via /api/live/<cam>/hls)."
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


def _frigate_base(request: Request):
    return pconf.get(_cfg(request), "FRIGATE_API", "http://frigate:5000")


# --------------------------------------------------------------------------
# app / static
# --------------------------------------------------------------------------
# index.html carries a {{ APP_VERSION }} token as the fallback text of the
# bottom-most version label. This route substitutes APP_VERSION (the single
# source of truth above) on every request, so the label always matches the
# running build without a hard-coded copy in the static HTML to keep in sync.
def _render_index() -> str:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return html.replace("{{ APP_VERSION }}", APP_VERSION)


@app.get("/", include_in_schema=False)
async def index():
    # no-cache: the page carries the live APP_VERSION - always revalidate so a
    # bumped version is shown immediately instead of a stale cached copy.
    return HTMLResponse(
        content=_render_index(),
        headers={"Cache-Control": "no-cache"},
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
# Live view: same-origin HLS proxy to Frigate's embedded go2rtc.
#
# go2rtc 1.9.10 (embedded in this Frigate 0.17.2 build) exposes NO working
# MSE-over-WebSocket for a generic client and no HTTP /api/stream.mse (404);
# its reliable, tunnel-friendly (plain TCP) live transports are HLS and MP4,
# both served under Frigate's /api/go2rtc/* reverse proxy. The SPA plays HLS
# with hls.js (native MSE in the page); any failure falls back to the detect-
# snapshot proxy. Routes below keep the whole HLS tree SAME-ORIGIN behind the
# session cookie:
#   GET /api/live/<cam>/hls/stream.m3u8          -> go2rtc master playlist
#   GET /api/live/<cam>/hls/hls/<playlist|seg>.ts -> go2rtc media/segments
# go2rtc's master references "hls/playlist.m3u8" relative to /api/, so the
# browser resolves it under the /hls/ mount here and no playlist rewriting is
# needed. go2rtc pulls the camera only while a viewer keeps fetching.
#
# AUDIO: the cameras send G.711 (PCMU) audio, which browsers/MSE cannot
# decode, so the master playlist is fetched for the <cam>_portal go2rtc
# source (config/config.yaml go2rtc.streams) - an on-demand ffmpeg transcode
# that copies the video and re-encodes the audio to AAC. hls.js/Safari can
# then play Live WITH sound; the SPA's Live stage exposes a muted-by-default
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
