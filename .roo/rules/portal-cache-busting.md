# portal cache-busting (versioning + fingerprinted static assets)

> **Exception — `/sw.js` (notification-only service worker).** The portal now
> ships one service worker at the STABLE, unfingerprinted URL `/sw.js` (served
> `Cache-Control: no-cache` by `portal/app.py`). A service worker registration
> REQUIRES a stable URL: a hashed URL would register a NEW worker on every
> release and break the update flow, so fingerprinting it is actively wrong.
> `/sw.js` has NO `fetch`/cache handler, so it cannot serve or shadow any asset
> and is otherwise outside this policy. See `plans/portal-notifications.md`.

The portal SPA (`portal/`) shows a STALE cached version when browsers cache the
JS/CSS under an unchanged URL. Enforced policy below, implemented in
`portal/app.py` (`_build_asset_manifest` / `static_asset`) and `portal/static/index.html`.

## Rules - follow on EVERY portal change

1. **Always bump the version.** Edit `APP_VERSION` in `portal/app.py` (single
   source of truth, `vX.Y.Z`, incremented on each task/update). It feeds the
   footer label, `/api/settings -> app_version`, AND the asset fingerprints.
   Never keep another copy of the number in the front-end.

2. **Never hard-code cacheable asset URLs.** In `portal/static/index.html`
   reference static files ONLY through the `{{ ASSET_* }}` tokens:
   - `{{ ASSET_STYLE }}`    -> `/static/style-<hash>.css`
   - `{{ ASSET_APP }}`      -> `/static/app-<hash>.js`
   - `{{ ASSET_HLS }}`      -> `/static/vendor/hls.min-<hash>.js`
   - `{{ ASSET_FAVICON }}`  -> `/static/favicon-<hash>.svg`
   - `{{ ASSET_ICON_192 }}` -> `/static/icons/icon-192-<hash>.png`  (PWA)
   - `{{ ASSET_ICON_512 }}` -> `/static/icons/icon-512-<hash>.png`  (PWA)
   - `{{ ASSET_ICON_MASKABLE }}` -> `/static/icons/icon-maskable-512-<hash>.png`
   - `{{ ASSET_ICON_APPLE }}` -> `/static/icons/apple-touch-icon-<hash>.png`
   The server substitutes them (see `_render_index()` in `portal/app.py`) with
   the CURRENT fingerprinted names, e.g. `/static/app-154kuhn7.js`.

   The icons are the installable-PWA set (`plans/portal-android-pwa.md`),
   generated from `portal/static/favicon.svg` by
   `dev_scripts/make_portal_pwa_icons.sh`. The PWA manifest is **not** a static
   file: `/manifest.webmanifest` is built PER REQUEST in `portal/app.py` from
   `_ASSET_FINGERPRINTS` (via `_icon_url()`) because only `index.html` gets
   token substitution - so always take icon URLs from that route, never hard-code
   them in the manifest either. There is deliberately **no service worker**.

3. **The fingerprint is automatic - do not hand-roll "random" names.** Each
   cacheable file gets an 8-char base36 suffix (like `154kuhn7`) = SHA-256 of
   (`APP_VERSION` + file bytes). So bumping `APP_VERSION` OR editing the file
   changes the served filename; browsers are then forced to load the new
   version. If you ADD a new cacheable static file (css/js/svg), add it to
   `CACHEABLE_ASSETS` in `portal/app.py` and use a new `{{ ASSET_* }}` token in
   `index.html`.

4. **Cache headers are set by the app.** Fingerprinted assets are served
   `Cache-Control: public, max-age=31536000, immutable` (safe - the URL changes
   on any content/version change). The page itself (`/`) is `no-cache` so it
   always re-fetches and picks up fresh fingerprinted URLs. Do not add
   cache-control meta tags or version query strings (`?v=`) in the HTML.

5. Per `.roo/rules/Agents.md`: create/update the plan file and `git commit`
   after the change. Deploy = user step (`git push` -> host pull ->
   `docker compose restart portal`).
