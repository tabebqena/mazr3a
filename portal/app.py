"""FastAPI app for the portal: auth + proxied Frigate JSON/media + Firewatch
evidence, plus the static SPA. See portal/__init__.py and plans/portal-web-app.md.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import (
    FileResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from portal import auth, config as pconf, firestore, frigate

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONF_PATH = os.environ.get("PORTAL_CONF", "/config/portal.conf")
COOKIE_NAME = "portal_session"
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
        "events/media, serves Firewatch fire evidence, and returns direct go2rtc "
        "MSE live-stream URLs (Access-gated at the Cloudflare Tunnel)."
    ),
    version="0.1.0",
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
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


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
        "stream_idle_timeout_s": pconf.geti(cfg, "STREAM_IDLE_TIMEOUT_S", 300),
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


@app.get("/api/stream-url/{camera}")
async def stream_url(camera: str, request: Request,
                     user: dict = Depends(current_user)):
    """Return the direct go2rtc MSE (WebSocket) URL for a camera.

    Frigate 0.17 serves go2rtc MSE over a WebSocket at .../live/mse/api/ws?
    src=<cam>. The SPA player swaps http->ws and connects; the URL is gated by
    the Cloudflare Access policy (public) or reachable on the LAN.
    """
    if not frigate.valid_camera_name(camera):
        raise HTTPException(status_code=400, detail="invalid camera name")
    tmpl = pconf.get(_cfg(request), "LIVE_URL_TMPL", "")
    if not tmpl or "{camera}" not in tmpl:
        raise HTTPException(status_code=503,
                            detail="LIVE_URL_TMPL not configured")
    return {"camera": camera, "url": tmpl.replace("{camera}", camera)}


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
