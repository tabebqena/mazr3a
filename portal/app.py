"""FastAPI app for the portal: auth + proxied Frigate JSON/media + Firewatch
evidence, plus the static SPA. See portal/__init__.py and plans/portal-web-app.md.
"""
import asyncio
import contextlib
import hashlib
import os
import time
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

from portal import (
    auth, config as pconf, eventstore, firestore, frigate, notifstore, usage,
    userstore,
)

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
APP_VERSION = "0.11.2"
_HTMX = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg, seed_users = pconf.parse_file(CONF_PATH)
    app.state.cfg = cfg
    app.state.secret = pconf.get(cfg, "SECRET_KEY")
    # Runtime user accounts: the portal OWNS media/portal/users.db (portal/
    # userstore.py), seeded ONCE from the `user` lines still in portal.conf.
    # The live store is read on every request, so create/edit/delete applies with
    # NO container restart (the old code snapshotted portal.conf users here).
    app.state.users_store = userstore.UserStore(
        pconf.get(cfg, "PORTAL_USERS_DB", "/media/portal/users.db"))
    try:
        app.state.users_store.configure()
        if seed_users:
            seeded = app.state.users_store.seed(seed_users)
            if seeded:
                print(f"user store: seeded {seeded} account(s) from portal.conf")
    except Exception as exc:
        print(f"WARNING: user store unavailable ({exc}) - login is impossible")
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        follow_redirects=True,
    )
    # Per-user bandwidth accounting (portal/usage.py). The portal is the only
    # writer of this DB; a unusable path degrades to a warning (playback must
    # never depend on stats). The background task flushes the in-memory
    # accumulator periodically and prunes rows past the retention window.
    app.state.usage = usage.UsageStore(
        pconf.get(cfg, "PORTAL_USAGE_DB", "/media/portal/usage.db"),
        retention_days=pconf.geti(cfg, "USAGE_RETENTION_DAYS", 180),
    )
    try:
        app.state.usage.configure()
    except Exception as exc:
        print(f"WARNING: usage store unavailable ({exc}) - "
              "bandwidth will not be persisted")
    usage_task = asyncio.create_task(app.state.usage.run())
    # Notification feed (portal/notifstore.py): the portal OWNS this DB too. It
    # holds the notification timeline + per-user read markers; a background
    # watcher (below) turns NEW Firewatch alerts into feed items. An unusable
    # path degrades to a warning - login/playback never depend on notifications.
    app.state.notifications = notifstore.NotificationStore(
        pconf.get(cfg, "PORTAL_NOTIF_DB", "/media/portal/notifications.db"),
        retention_days=pconf.geti(cfg, "NOTIFY_RETENTION_DAYS", 90))
    try:
        app.state.notifications.configure()
    except Exception as exc:
        print(f"WARNING: notification store unavailable ({exc}) - "
              "the notification feed will be empty")
    notif_task = asyncio.create_task(_notification_watcher(app))
    if not app.state.secret:
        print("WARNING: portal.conf SECRET_KEY is empty - sessions will not work")
    if not app.state.users_store.count():
        print("WARNING: no users configured - login is impossible")
    yield
    usage_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await usage_task
    notif_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await notif_task
    app.state.usage.flush()
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
    """Dependency: the authenticated user dict or a 401.

    Reads the LIVE user store (portal/userstore.py), so a deleted account's
    session is rejected on the next request - no restart needed.
    """
    if not request.app.state.secret:
        raise HTTPException(status_code=401, detail="not configured")
    token = request.cookies.get(COOKIE_NAME)
    name = auth.read_session(request.app.state.secret, token)
    if not name:
        raise HTTPException(status_code=401, detail="not authenticated")
    user = request.app.state.users_store.get(name)
    if user is None or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


def current_admin(request: Request, user: dict = Depends(current_user)):
    """Dependency: like current_user but ONLY an admin (403 otherwise).

    `is_admin` is a column in the runtime users DB and is editable from the
    Account tab, so the front-end nav gating uses /api/me is_admin AND this
    server-side check is the real gate for admin-only routes.
    """
    if not user.get("is_admin"):
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
# The icons/*.png are the installable-PWA icon set (Android home-screen icon),
# generated from favicon.svg by dev_scripts/make_portal_pwa_icons.sh and
# referenced by /manifest.webmanifest (see _icon_url below) - fingerprinting
# them means bumping APP_VERSION or editing an icon rotates its URL.
CACHEABLE_ASSETS = [
    ("style.css", "ASSET_STYLE"),
    ("app.js", "ASSET_APP"),
    ("vendor/hls.min.js", "ASSET_HLS"),
    ("favicon.svg", "ASSET_FAVICON"),
    ("icons/icon-192.png", "ASSET_ICON_192"),
    ("icons/icon-512.png", "ASSET_ICON_512"),
    ("icons/icon-maskable-512.png", "ASSET_ICON_MASKABLE"),
    ("icons/apple-touch-icon.png", "ASSET_ICON_APPLE"),
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


def _icon_url(token: str) -> str:
    """Fingerprinted public URL of a PWA icon, e.g. /static/icons/icon-512-<h>.png."""
    return f"/static/{_ASSET_FINGERPRINTS[token]}"


# Installable-PWA manifest (Android: Chrome -> "Install app" -> WebAPK). Built
# PER REQUEST rather than served as a static file because its icon URLs must
# carry the CURRENT fingerprints - _render_index() only rewrites index.html, so
# a static manifest could not. no-cache so a bumped APP_VERSION (or an edited
# icon) is reflected immediately.
@app.get("/manifest.webmanifest", include_in_schema=False)
async def manifest():
    doc = {
        "id": "/",
        "name": "mazr3a CCTV",
        "short_name": "mazr3a",
        "description": "Farm CCTV portal - live view, events and fire alerts",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "any",
        "background_color": "#0b0e12",
        "theme_color": "#0b0e12",
        "icons": [
            {"src": _icon_url("ASSET_ICON_192"), "sizes": "192x192",
             "type": "image/png", "purpose": "any"},
            {"src": _icon_url("ASSET_ICON_512"), "sizes": "512x512",
             "type": "image/png", "purpose": "any"},
            {"src": _icon_url("ASSET_ICON_MASKABLE"), "sizes": "512x512",
             "type": "image/png", "purpose": "maskable"},
        ],
    }
    return JSONResponse(
        doc,
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache"},
    )


# iOS probes /apple-touch-icon.png by default; the fingerprinted icon is linked
# from index.html, this is the convenience default (current-but-no-cache, since
# the bare name is not fingerprinted).
@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_touch_icon():
    return FileResponse(
        STATIC_DIR / "icons/apple-touch-icon.png",
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


# Notification-only service worker (portal/static/sw.js). Android Chrome cannot
# construct `new Notification()` (it throws `Illegal constructor`), so browser
# notifications MUST go through ServiceWorkerRegistration.showNotification().
# The worker only handles `notificationclick` (focus/open the SPA deep link) and
# has NO fetch/cache handler, so it cannot influence asset caching or the
# fingerprint policy. Deliberately NOT fingerprinted: a service worker
# registration needs a STABLE URL (a hashed URL would register a new worker on
# every release and break the update flow); served no-cache so a change is seen
# on the next update check. See plans/portal-notifications.md and
# .roo/rules/portal-cache-busting.md.
@app.get("/sw.js", include_in_schema=False)
async def service_worker():
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------
@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    username = str(body.get("username") or "").strip()
    password = str(body.get("password") or "")
    # Live store lookup (usernames are case-insensitive): a user created from
    # the Account tab can sign in immediately, with no container restart. An
    # INACTIVE account can never sign in.
    user = request.app.state.users_store.get(username)
    if (user and user.get("is_active")
            and auth.verify_password(password, user["password_hash"])):
        response = JSONResponse({"ok": True, "username": user["username"],
                                 "display_name": user["display_name"],
                                 "default_camera": user["default_camera"]})
        _set_session(response, user["username"], request)
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
        "display_name": user.get("display_name") or user["username"],
        "photo": user.get("photo") or "",
        "is_admin": bool(user.get("is_admin")),
        "permissions": _effective_permissions(user),
        "quota_bytes": int(user.get("quota_bytes") or 0),
        "default_camera": (user.get("default_camera")
                           or pconf.get(_cfg(request), "DEFAULT_CAMERA")),
    }


# --------------------------------------------------------------------------
# User management - RUNTIME accounts, NO container restart.
#
# Accounts live in the portal's own SQLite DB (portal/userstore.py) so they can
# be created / edited / deleted from the Account tab at once (previously they
# were `user` lines in portal.conf, read once at startup). Those lines are now
# only a first-run SEED.
#
# Self-service: a user may edit their own display name / photo / default camera
# and change their own password. Admin-only: create/delete users, toggle
# is_admin / is_active, set permissions + quota, inspect the stored
# password_hash. The last ACTIVE administrator can never be demoted,
# deactivated or deleted, and nobody can delete/deactivate their own account.
# --------------------------------------------------------------------------
def _is_admin(user: dict) -> bool:
    return bool(user and user.get("is_admin"))


def _effective_permissions(user: dict) -> list:
    """The `tab_*` keys a user may open.

    An ADMIN implicitly has EVERY tab - the full CURRENT catalogue is reported
    so the nav shows all of them, and `is_admin` keeps any FUTURE tab open
    automatically. A non-admin gets exactly the stored keys (EMPTY = no tabs).
    """
    if _is_admin(user):
        return list(userstore.ALLOWED_PERMISSIONS)
    return list(user.get("permissions") or [])


def _same_user(user: dict, username: str) -> bool:
    return (user.get("username") or "").lower() == (username or "").lower()


def _user_store(request: Request) -> userstore.UserStore:
    return request.app.state.users_store


def _user_or_404(request: Request, username: str) -> dict:
    record = _user_store(request).get_public(username)
    if record is None:
        raise HTTPException(status_code=404, detail="user not found")
    return record


@app.get("/api/users")
async def list_users(request: Request, admin: dict = Depends(current_admin)):
    """All accounts (admin only) + the assignable permission catalogue.

    Includes each stored password_hash so the admin Users table can show it.
    """
    return {"users": _user_store(request).list(include_secrets=True),
            "permissions": list(userstore.ALLOWED_PERMISSIONS),
            "tabs": list(userstore.AVAILABLE_TABS)}


@app.post("/api/users")
async def create_user(request: Request, admin: dict = Depends(current_admin)):
    """Create an account (admin only). Applies immediately - no restart."""
    body = await request.json()
    try:
        username = userstore.normalize_username(body.get("username"))
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid username")
    password = str(body.get("password") or "")
    if len(password) < 6:
        raise HTTPException(status_code=400,
                            detail="password must be at least 6 characters")
    if not userstore.valid_photo(body.get("photo")):
        raise HTTPException(status_code=400, detail="invalid photo")
    store = _user_store(request)
    if store.get(username) is not None:
        raise HTTPException(status_code=409, detail="username already exists")
    try:
        store.create(
            username, auth.hash_password(password),
            display_name=body.get("display_name") or username,
            is_admin=bool(body.get("is_admin")),
            is_active=bool(body.get("is_active", True)),
            permissions=body.get("permissions"),
            # Omitted (None) -> the users table column DEFAULT (5 GiB); an
            # explicit 0 = unlimited.
            quota_bytes=body.get("quota_bytes"),
            default_camera=body.get("default_camera") or "",
        )
        if body.get("photo"):
            store.update(username, photo=body["photo"])
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"could not create user: {exc}")
    return store.get_public(username)


@app.get("/api/users/{username}")
async def get_user(username: str, request: Request,
                   user: dict = Depends(current_user)):
    """One account: your own, or any account for an admin."""
    if not (_is_admin(user) or _same_user(user, username)):
        raise HTTPException(status_code=403, detail="not allowed")
    return _user_or_404(request, username)


@app.patch("/api/users/{username}")
async def update_user(username: str, request: Request,
                      user: dict = Depends(current_user)):
    """Update an account: self (display name/photo/default camera) or admin."""
    store = _user_store(request)
    target = store.get(username)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    admin = _is_admin(user)
    if not (admin or _same_user(user, target["username"])):
        raise HTTPException(status_code=403, detail="not allowed")
    body = await request.json()
    fields = {}
    if "display_name" in body:
        fields["display_name"] = str(body.get("display_name") or "")[:64]
    if "default_camera" in body:
        fields["default_camera"] = str(body.get("default_camera") or "")[:64]
    if "photo" in body:
        if not userstore.valid_photo(body.get("photo")):
            raise HTTPException(status_code=400, detail="invalid photo")
        fields["photo"] = str(body.get("photo") or "")
    if admin:
        if "permissions" in body:
            fields["permissions"] = userstore.normalize_permissions(
                body.get("permissions"))
        if "quota_bytes" in body:
            try:
                fields["quota_bytes"] = max(0, int(body.get("quota_bytes") or 0))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="invalid quota")
        if "is_admin" in body:
            new_admin = 1 if body.get("is_admin") else 0
            # An admin must never change their OWN administrator flag from the
            # portal (the Manage tab disables the checkbox too) - it is far too
            # easy to lock yourself out by mistake.
            if not new_admin and _same_user(user, target["username"]):
                raise HTTPException(
                    status_code=400,
                    detail="cannot change your own administrator access")
            # Never remove the last USABLE administrator (would lock everyone
            # out). An already-inactive admin does not count.
            if (not new_admin and target["is_admin"] and target["is_active"]
                    and store.count_admins() <= 1):
                raise HTTPException(
                    status_code=400,
                    detail="cannot remove the last administrator")
            fields["is_admin"] = new_admin
        if "is_active" in body:
            new_active = 1 if body.get("is_active") else 0
            if not new_active and _same_user(user, target["username"]):
                raise HTTPException(
                    status_code=400,
                    detail="cannot deactivate your own account")
            # Never disable the last usable administrator.
            if (not new_active and target["is_active"] and target["is_admin"]
                    and store.count_admins() <= 1):
                raise HTTPException(
                    status_code=400,
                    detail="cannot deactivate the last administrator")
            fields["is_active"] = new_active
    try:
        store.update(target["username"], **fields)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return store.get_public(target["username"])


@app.post("/api/users/{username}/password")
async def change_user_password(username: str, request: Request,
                               user: dict = Depends(current_user)):
    """Change a password: self (must confirm current) or an admin reset."""
    store = _user_store(request)
    target = store.get(username)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    is_self = _same_user(user, target["username"])
    if not (_is_admin(user) or is_self):
        raise HTTPException(status_code=403, detail="not allowed")
    body = await request.json()
    new_password = str(body.get("new_password") or "")
    if len(new_password) < 6:
        raise HTTPException(status_code=400,
                            detail="password must be at least 6 characters")
    # Anyone changing THEIR OWN password must prove the current one; an admin
    # resetting ANOTHER user's password does not need it.
    if is_self and not auth.verify_password(
            str(body.get("current_password") or ""), target["password_hash"]):
        raise HTTPException(status_code=403,
                            detail="current password is incorrect")
    store.set_password(target["username"], auth.hash_password(new_password))
    return {"ok": True}


@app.delete("/api/users/{username}")
async def delete_user(username: str, request: Request,
                      admin: dict = Depends(current_admin)):
    """Delete an account (admin only). Never your own or the last admin."""
    store = _user_store(request)
    target = store.get(username)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    if _same_user(admin, target["username"]):
        raise HTTPException(status_code=400,
                            detail="cannot delete your own account")
    if (target["is_admin"] and target["is_active"]
            and store.count_admins() <= 1):
        raise HTTPException(status_code=400,
                            detail="cannot delete the last administrator")
    store.delete(target["username"])
    return {"ok": True}


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


# Egress byte kinds recorded by the usage store (portal/usage.py).
#   live   = live video transports (MSE WebSocket + HLS playlist/segments)
#   events = event video clips (/api/events/<id>/clip.mp4)
#   other  = snapshots (live/event) and the events JSON listing
USAGE_LIVE = "live"
USAGE_EVENTS = "events"
USAGE_OTHER = "other"


def _count_egress(request: Request, username: str, kind: str, nbytes: int):
    """Best-effort: record `nbytes` sent to `username`. Never raises.

    Bandwidth accounting must never affect playback, so any failure here is
    swallowed - the streaming path is the critical path.
    """
    if not nbytes or not username or not kind:
        return
    try:
        request.app.state.usage.add(username, kind, nbytes)
    except Exception:
        pass


async def _frigate_stream(request: Request, path: str, params=None,
                          username: str = "", kind: str = ""):
    """Stream an upstream Frigate route verbatim (content-type preserved).

    go2rtc is embedded in Frigate and its HTTP API is reverse-proxied by
    Frigate under /api/go2rtc/* on port 5000. Requests here carry the session
    cookie (same-origin) and are relayed server-side, so the go2rtc HLS tree
    never leaves the portal origin / Cloudflare Access boundary.

    Egress: the bytes streamed to the browser are counted (kind) for the
    per-user bandwidth usage store - live HLS goes to USAGE_LIVE.
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
        sent = 0
        try:
            async for chunk in resp.aiter_bytes(64 * 1024):
                sent += len(chunk)
                yield chunk
        finally:
            await resp.aclose()
            _count_egress(request, username, kind, sent)

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
        params={"src": camera + LIVE_SOURCE_SUFFIX},
        username=user["username"], kind=USAGE_LIVE)


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
                                 params=request.query_params,
                                 username=user["username"], kind=USAGE_LIVE)


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
    ws_user = websocket.app.state.users_store.get(name) if name else None
    authed = bool(ws_user) and bool(ws_user.get("is_active"))
    if not authed or not frigate.valid_camera_name(camera):
        await websocket.close(code=4401)
        return
    usage_store = websocket.app.state.usage

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
        sent = 0
        try:
            async for message in upstream:
                if isinstance(message, (bytes, bytearray)):
                    await websocket.send_bytes(bytes(message))
                else:
                    await websocket.send_text(message)
                sent += len(message)
        except Exception:
            pass
        finally:
            # Best-effort egress accounting; `name` is the authenticated
            # username (validated above). A stats failure must never affect
            # the relay.
            if sent and name:
                try:
                    usage_store.add(name, USAGE_LIVE, sent)
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
                                "image/jpeg", username=user["username"],
                                kind=USAGE_OTHER)


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
    # The events JSON listing is small but real egress - count it as `other`.
    _count_egress(request, user["username"], USAGE_OTHER, len(resp.content))
    return JSONResponse(content=resp.json())


# --------------------------------------------------------------------------
# Frigate media proxy (event snapshot / clip) - streamed, gated by the portal
# cookie (the SPA is same-origin so the cookie flows on <img>/fetch too).
# --------------------------------------------------------------------------
async def _frigate_media(request: Request, path: str, media_type: str,
                         username: str = "", kind: str = ""):
    """Stream a Frigate snapshot/clip through the portal cookie.

    Range is FORWARDED so a <video> can seek (the large Events player relies on
    this): the browser's Range/If-Range go upstream and a 206 is relayed with
    Content-Range/Accept-Ranges. Without this the proxy always returned a full
    200 and the player could not scrub.

    Egress: the bytes relayed to the browser are counted (kind) per username;
    bytes re-sent for a Range seek are counted too (they are real egress).
    """
    client = request.app.state.client
    fwd = {}
    if request.headers.get("range"):
        fwd["Range"] = request.headers["range"]
    if request.headers.get("if-range"):
        fwd["If-Range"] = request.headers["if-range"]
    req = client.build_request("GET", _frigate_base(request) + path,
                               params=request.query_params, headers=fwd)
    try:
        resp = await client.send(req, stream=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"frigate unreachable: {exc}")
    if resp.status_code not in (200, 206):
        await resp.aclose()
        raise HTTPException(status_code=resp.status_code, detail="frigate error")

    async def gen():
        sent = 0
        try:
            async for chunk in resp.aiter_bytes(64 * 1024):
                sent += len(chunk)
                yield chunk
        finally:
            await resp.aclose()
            _count_egress(request, username, kind, sent)

    headers = {"Cache-Control": "no-store", "Accept-Ranges": "bytes"}
    for h in ("content-range", "content-length"):
        if resp.headers.get(h):
            headers[h] = resp.headers[h]
    return StreamingResponse(gen(), status_code=resp.status_code,
                             media_type=media_type, headers=headers)


@app.get("/api/events/{event_id}/snapshot.jpg")
async def event_snapshot(event_id: str, request: Request,
                          user: dict = Depends(current_user)):
    return await _frigate_media(
        request, f"/api/events/{event_id}/snapshot.jpg", "image/jpeg",
        username=user["username"], kind=USAGE_OTHER)


@app.get("/api/events/{event_id}/clip.mp4")
async def event_clip(event_id: str, request: Request,
                     user: dict = Depends(current_user)):
    return await _frigate_media(
        request, f"/api/events/{event_id}/clip.mp4", "video/mp4",
        username=user["username"], kind=USAGE_EVENTS)


# --------------------------------------------------------------------------
# Bandwidth usage - the real egress the portal sent to each logged-in user.
#
# Counted at the proxy/relay chokepoints above (live MSE/HLS, event clips,
# snapshots, events JSON) and persisted by portal/usage.py as DAILY counters
# keyed by user + day + kind. `/api/usage` is the CALLER's own breakdown;
# `/api/admin/usage` is every user (admin only, same gate as the Debug tab).
# --------------------------------------------------------------------------
@app.get("/api/usage")
async def my_usage(request: Request, user: dict = Depends(current_user)):
    """The caller's own bandwidth breakdown (today / 7d / 30d / total)."""
    return request.app.state.usage.totals(user["username"])


@app.get("/api/admin/usage")
async def admin_usage(request: Request, admin: dict = Depends(current_admin)):
    """Every user's bandwidth breakdown, biggest consumer first (admin only)."""
    return request.app.state.usage.all_users()


# --------------------------------------------------------------------------
# Notifications - the portal's own feed (portal/notifstore.py).
#
# The feed is a GLOBAL timeline; only the READ position is per-user. New items
# are raised by the SPA as browser notifications (it polls ?after_id=). The
# watcher below fills the feed from NEW Firewatch alerts; an admin can also
# publish a system message. See plans/portal-notifications.md.
# --------------------------------------------------------------------------
@app.get("/api/notifications")
async def notifications_list(request: Request,
                             user: dict = Depends(current_user),
                             limit: int = 50, offset: int = 0,
                             after_id: Optional[int] = None):
    """The caller's notification feed.

    Without `after_id`: the newest-first PAGE for the Notifications tab.
    With `after_id`: only the NEWER items, ASCENDING - the client poller uses
    this to raise one browser notification per new item. Both shapes carry
    `unread`, `total` and `latest_id`.
    """
    return request.app.state.notifications.list_for(
        user["username"], limit=limit, offset=offset, after_id=after_id)


@app.post("/api/notifications/read")
async def notifications_read(request: Request,
                             user: dict = Depends(current_user)):
    """Mark notifications read: `{ids:[...]}` or `{all:true}` -> `{unread}`."""
    body = await request.json()
    store = request.app.state.notifications
    if body.get("all"):
        return {"unread": store.mark_all_read(user["username"])}
    return {"unread": store.mark_read(user["username"], body.get("ids") or [])}


@app.post("/api/admin/notifications")
async def admin_publish_notification(request: Request,
                                     admin: dict = Depends(current_admin)):
    """Publish a SYSTEM notification (admin only)."""
    body = await request.json()
    title = str(body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title required")
    item = request.app.state.notifications.add(
        kind=str(body.get("kind") or "system"),
        title=title,
        body=str(body.get("body") or ""),
        url=str(body.get("url") or "") or "#/notifications",
        camera=str(body.get("camera") or ""))
    if item is None:
        raise HTTPException(status_code=503,
                            detail="could not store notification")
    return item


# --------------------------------------------------------------------------
# Notification ingestion - Firewatch fire alerts -> the portal feed.
#
# The portal already reads the firewatch evidence DB (firestore.py, read-only).
# This watcher polls it for NEW `alerted=1` frames and records one notification
# each (dedup_key `fire:<id>`) so the SPA can raise a browser notification and
# the Notifications tab keeps the history. The high-water frame id lives in the
# notifstore meta table; the FIRST run BASELINES to the current max so a fresh
# install never notifies the whole historical alert backlog.
# --------------------------------------------------------------------------
def _scan_fire_alerts(store, db):
    raw = store.get_meta("fire_last_frame_id")
    if raw is None:
        store.set_meta("fire_last_frame_id", str(firestore.max_alerted_id(db)))
        return
    try:
        last = int(raw)
    except (TypeError, ValueError):
        last = 0
    rows = firestore.alerted_since(db, last, limit=50)
    if not rows:
        return
    newest = last
    for row in rows:
        try:
            stamp = float(row.get("captured_at") or 0)
        except (TypeError, ValueError):
            stamp = 0.0
        # One alert per CAMERA per 3-minute burst. firewatch stores a dense run
        # of follow-up samples for a single incident, so keying on the frame id
        # alone would spam the user with one notification per sample; the
        # partial unique index on dedup_key turns repeats into no-ops.
        bucket = int(stamp // 180) if stamp else 0
        shown = str(row.get("ts_utc") or "")
        body = "score {:.2f}".format(float(row.get("best_score") or 0))
        if shown:
            body += " \u00b7 " + shown
        store.add("fire",
                  title="Fire alert - " + (row.get("camera") or "camera"),
                  body=body,
                  url="#/fire",
                  camera=row.get("camera") or "",
                  dedup_key="fire:%s:%d" % (row.get("camera") or "", bucket),
                  created_at=stamp or None)
        try:
            newest = max(newest, int(row.get("id") or 0))
        except (TypeError, ValueError):
            pass
    store.set_meta("fire_last_frame_id", str(newest))


async def _notification_watcher(app: FastAPI):
    """Poll the firewatch DB for new alerts; prune the feed once a day."""
    interval = max(5, pconf.geti(app.state.cfg, "NOTIFY_POLL_S", 30))
    last_prune = 0.0
    while True:
        try:
            await asyncio.sleep(interval)
            db = pconf.get(app.state.cfg, "FIREWATCH_DB", "/media/firewatch.db")
            _scan_fire_alerts(app.state.notifications, db)
            now = time.time()
            if now - last_prune >= 86400:
                last_prune = now
                app.state.notifications.prune()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


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


@app.get("/api/scenereader/scenes")
async def scenereader_scenes(request: Request, user: dict = Depends(current_user),
                             day: Optional[str] = None, camera: Optional[str] = None,
                             after: Optional[float] = None, before: Optional[float] = None,
                             with_events: Optional[int] = None,
                             limit: int = 100, offset: int = 0):
    """The adaptive scenes (L1): one camera's burst of activity, the objects that
    COEXISTED in it, and which of them actually MOVED.

    Deliberately under /api/scenereader/* so it never collides with the legacy
    /api/scenes compat shim below (a different shape over a different grain).
    """
    db, _status, _trigger = _scenereader_paths(request)
    result = eventstore.list_scenes(
        db, day=day or None, camera=camera or None, after=after, before=before,
        with_events=bool(with_events), limit=limit, offset=offset)
    result["days"] = eventstore.scene_days(db)
    return result


@app.get("/api/scenereader/scenes/{scene_id}")
async def scenereader_scene_detail(scene_id: int, request: Request,
                                   user: dict = Depends(current_user)):
    """One scene WITH its member captures (each proxied to a Frigate snapshot)."""
    db, _status, _trigger = _scenereader_paths(request)
    item = eventstore.get_scene(db, scene_id)
    if item is None:
        raise HTTPException(status_code=404, detail="scene not found")
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
