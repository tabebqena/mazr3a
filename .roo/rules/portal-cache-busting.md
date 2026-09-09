# portal cache-busting (versioning + fingerprinted static assets)

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
   - `{{ ASSET_STYLE }}`   -> `/static/style-<hash>.css`
   - `{{ ASSET_APP }}`     -> `/static/app-<hash>.js`
   - `{{ ASSET_HLS }}`     -> `/static/vendor/hls.min-<hash>.js`
   - `{{ ASSET_FAVICON }}` -> `/static/favicon-<hash>.svg`
   The server substitutes them (see `_render_index()` in `portal/app.py`) with
   the CURRENT fingerprinted names, e.g. `/static/app-154kuhn7.js`.

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
