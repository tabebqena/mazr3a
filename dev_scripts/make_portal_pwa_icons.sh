#!/usr/bin/env bash
# make_portal_pwa_icons.sh - rasterize the portal's SINGLE brand source
# (portal/static/favicon.svg) into the installable-PWA icon set under
# portal/static/icons/.
#
# ONE BRAND SOURCE: portal/static/favicon.svg. Every PNG below is GENERATED -
# never hand-edit them. Re-run this script after changing the favicon so the
# home-screen icon stays in sync with the tab icon.
#
# Renderer preference (the FIRST available one is used):
#   1. rsvg-convert  (librsvg)          - best fidelity, no browser needed
#   2. inkscape                         - --export-type=png
#   3. google-chrome / chromium         - headless render; matches how Android
#                                         itself rasterizes the brand
#
# ImageMagick is deliberately NOT used for SVG: its built-in MSVG renderer
# silently DROPS fills (verified - the green #35c26f lens renders black), so
# `convert favicon.svg` produces a wrong icon. ImageMagick would only be safe
# for PNG compositing, which this script does not need.
#
# Usage:  dev_scripts/make_portal_pwa_icons.sh
# Output: portal/static/icons/{icon-192.png,icon-512.png,icon-maskable-512.png,
#         apple-touch-icon.png}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_SVG="${REPO_ROOT}/portal/static/favicon.svg"
OUT_DIR="${REPO_ROOT}/portal/static/icons"
# The dark card colour used by favicon.svg (must match its outer <rect fill>).
BG="#0b0e12"

# Maskable icons are cropped to (at most) a centred circle of 80% diameter, so
# the mark is scaled down to sit safely inside it. 0.62 of the canvas is a
# comfortable margin (art must stay within ~40% of the radius from centre).
MASKABLE_SCALE="0.62"

if [[ ! -f "${SRC_SVG}" ]]; then
  echo "ERROR: brand source not found: ${SRC_SVG}" >&2
  exit 1
fi

# ---------------------------------------------------------------- renderer ---
RSVG=""; INKSCAPE=""; CHROME=""
if command -v rsvg-convert >/dev/null 2>&1; then
  RSVG="$(command -v rsvg-convert)"
elif command -v inkscape >/dev/null 2>&1; then
  INKSCAPE="$(command -v inkscape)"
elif command -v google-chrome >/dev/null 2>&1; then
  CHROME="$(command -v google-chrome)"
elif command -v google-chrome-stable >/dev/null 2>&1; then
  CHROME="$(command -v google-chrome-stable)"
elif command -v chromium >/dev/null 2>&1; then
  CHROME="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROME="$(command -v chromium-browser)"
else
  cat >&2 <<'EOF'
ERROR: no SVG rasterizer found. Install ONE of:
  - librsvg        (provides rsvg-convert)   <- preferred
  - inkscape
  - google-chrome / chromium                 <- used headless
EOF
  exit 1
fi

# Inner markup of favicon.svg = everything between the outer <svg ...> and
# </svg> tags (both on their own lines). Keeps ONE brand source.
inner_svg() {
  sed -n '/<svg[^>]*>/,/<\/svg>/p' "${SRC_SVG}" | sed '1d;$d'
}

# ---------------------------------------------------------------- variants ---
# Build a standalone variant SVG on stdout.
#   $1 = pixel size   $2 = "any" | "maskable" | "apple"
variant_svg() {
  local size="$1" kind="$2" bg_rect="" mark=""
  if [[ "${kind}" != "any" ]]; then
    # Full-bleed opaque background (maskable icons must NOT be transparent).
    bg_rect="<rect width='64' height='64' fill='${BG}'/>"
  fi
  if [[ "${kind}" == "maskable" ]]; then
    mark="<g transform='translate(32,32) scale(${MASKABLE_SCALE}) translate(-32,-32)'>$(inner_svg)</g>"
  else
    mark="$(inner_svg)"
  fi
  printf '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="%s" height="%s">%s%s</svg>\n' \
    "${size}" "${size}" "${bg_rect}" "${mark}"
}

# Rasterize a variant to a PNG of exactly ${size}x${size}.
render() {
  local kind="$1" size="$2" out="$3"
  local tmp_svg tmp_html
  tmp_svg="$(mktemp --suffix=.svg)"
  variant_svg "${size}" "${kind}" > "${tmp_svg}"

  if [[ -n "${RSVG}" ]]; then
    "${RSVG}" -w "${size}" -h "${size}" -b none -o "${out}" "${tmp_svg}"
  elif [[ -n "${INKSCAPE}" ]]; then
    "${INKSCAPE}" "${tmp_svg}" --export-type=png \
      --export-filename="${out}" \
      --export-width="${size}" --export-height="${size}" >/dev/null 2>&1
  else
    # Headless Chrome renders an image document with page margins/handles, so
    # inline the SVG in a zero-margin HTML page sized exactly to the viewport.
    tmp_html="$(mktemp --suffix=.html)"
    {
      printf '<!doctype html><html><head><meta charset="utf-8"><style>'
      printf 'html,body{margin:0;padding:0;background:transparent;overflow:hidden}'
      printf 'svg{display:block}</style></head><body>'
      cat "${tmp_svg}"
      printf '</body></html>'
    } > "${tmp_html}"
    "${CHROME}" --headless=new --no-sandbox --disable-gpu --hide-scrollbars \
      --force-device-scale-factor=1 --window-size="${size},${size}" \
      --default-background-color=00000000 \
      --screenshot="${out}" "file://${tmp_html}" >/dev/null 2>&1
    rm -f "${tmp_html}"
  fi
  rm -f "${tmp_svg}"

  if [[ ! -s "${out}" ]]; then
    echo "ERROR: failed to render ${out} (renderer: ${RSVG:-${INKSCAPE:-${CHROME}}})" >&2
    exit 1
  fi
}

# ------------------------------------------------------------------- build ---
mkdir -p "${OUT_DIR}"
echo "brand source : ${SRC_SVG}"
echo "renderer     : ${RSVG:-${INKSCAPE:-${CHROME}}}"
echo

render any      192 "${OUT_DIR}/icon-192.png"
render any      512 "${OUT_DIR}/icon-512.png"
render maskable 512 "${OUT_DIR}/icon-maskable-512.png"
render apple    180 "${OUT_DIR}/apple-touch-icon.png"

echo "generated:"
ls -l "${OUT_DIR}"/*.png
echo
echo "OK - remember to bump APP_VERSION in portal/app.py (the icon fingerprints"
echo "     derive from it), then commit the PNGs."
