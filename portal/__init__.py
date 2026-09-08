"""portal - authenticated web portal (FastAPI backend + vanilla JS SPA) for the
Frigate + Firewatch stack.

Backend API routes live under /api/* behind a signed HttpOnly session cookie;
Frigate REST/DB data is exposed only through those routes. The Live view is
DETECT-SNAPSHOT mode: the SPA polls GET /api/live/<cam>/latest.jpg (Frigate's
already-decoded detect frame, ~1 fps, no extra decode). MSE/go2rtc true
streaming is deferred (go2rtc has no camera streams configured on the host).
See plans/portal-web-app.md.
"""
