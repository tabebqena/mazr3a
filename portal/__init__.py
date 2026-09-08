"""portal - authenticated web portal (FastAPI backend + vanilla JS SPA) for the
Frigate + Firewatch stack.

Backend API routes live under /api/* behind a signed HttpOnly session cookie;
Frigate REST/DB data is exposed only through those routes. Live MSE media is
played by the SPA from a DIRECT go2rtc URL returned by GET /api/stream-url/<cam>
- that URL is gated at the edge by the Cloudflare Tunnel Access policy, not by
the portal cookie. See plans/portal-web-app.md.
"""
