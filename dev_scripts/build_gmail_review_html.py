#!/usr/bin/env python3
"""build_gmail_review_html.py - render the fire-alert review as a single HTML page.

Reads media/Gmail/fire_alerts_scores.csv (the review the user filled in:
file, alert time, printed score, V4 fire, V4 smoke, verdict, optional note) and
writes a self-contained page showing, for every alert:
  - the CCTV-frame image (cropped_frames/<name>.jpg; click opens the original)
  - the printed (old/alerting) score, e.g. "fire 0.77"
  - the ACTIVE model (v4) score on the frame (fire and smoke)
  - the user's verdict (TRUE fire / FALSE positive) + note
The page is grouped TRUE / FALSE with a summary header. It uses only relative
image paths + inline CSS so it works by double-clicking the file (no server).

Usage:  python3 dev_scripts/build_gmail_review_html.py [--src media/Gmail]
"""
import argparse
import csv
import html
import os
import re

SRC = os.path.join("media", "Gmail")
CROPS_DIR = "cropped_frames"
OUT = "fire_alerts_review.html"

VERDICT_RE = re.compile(r"^\s*(true|false)\b", re.IGNORECASE)

ESC = html.escape


def esc_num(v):
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    args = ap.parse_args()

    csv_path = os.path.join(args.src, "fire_alerts_scores.csv")
    with open(csv_path, encoding="utf-8") as fh:
        raw = list(csv.reader(fh))
    if not raw:
        raise SystemExit(f"empty csv: {csv_path}")
    header = raw[0]
    # tolerate trailing comma => a 7th, unnamed column holds the user's note
    has_note = len(header) >= 7 or any(len(r) >= 7 for r in raw[1:])

    cards = []
    for r in raw[1:]:
        if not r or not r[0].strip():
            continue
        row = (r + [""] * 7)[:7]
        fname, alert_ts, printed, v4f, v4s, verdict, note = row
        m = VERDICT_RE.match(verdict or "")
        is_true = bool(m and m.group(1).lower() == "true")
        cards.append({
            "file": fname,
            "alert_ts": alert_ts,
            "printed": printed or "",
            "v4_fire": esc_num(v4f),
            "v4_smoke": esc_num(v4s),
            "verdict": (m.group(1).upper() if m else (verdict or "").strip()),
            "is_true": is_true,
            "note": note.strip(),
            "crop": os.path.join(CROPS_DIR, fname),
        })

    trues = [c for c in cards if c["is_true"]]
    falses = [c for c in cards if not c["is_true"]]

    def score_cell(c):
        fire_alert = float(c["v4_fire"] or 0) >= 0.5
        smoke_alert = float(c["v4_smoke"] or 0) >= 0.5
        tag = ""
        if fire_alert and smoke_alert:
            tag = '<span class="tag both">V4 would flag: fire+smoke</span>'
        elif fire_alert:
            tag = '<span class="tag fire">V4 would flag: fire</span>'
        elif smoke_alert:
            tag = '<span class="tag smoke">V4 would flag: smoke</span>'
        return (f'<span class="kv"><span class="k">V4 fire</span>'
                f'<span class="v">{c["v4_fire"]}</span></span>'
                f'<span class="kv"><span class="k">V4 smoke</span>'
                f'<span class="v">{c["v4_smoke"]}</span></span>'
                f"<span class='break'>{tag}</span>")

    def card_html(c):
        badge = ('<span class="badge true">TRUE &#128293; fire</span>'
                 if c["is_true"] else
                 '<span class="badge false">FALSE &#10060;</span>')
        note = (f'<div class="note">{ESC(c["note"])}</div>' if c["note"] else "")
        return f"""
        <div class="card {'true' if c['is_true'] else 'false'}">
          <div class="imgwrap">
            <a href="{ESC(c['file'])}" target="_blank" title="Open original screenshot">
              <img loading="lazy" src="{ESC(c['crop'])}" alt="{ESC(c['file'])}">
            </a>
          </div>
          <div class="meta">
            <div class="file">{ESC(c['file'])}</div>
            <div class="row">
              <span class="kv"><span class="k">Alert time</span>
                <span class="v">{ESC(c['alert_ts'])}</span></span>
              <span class="kv"><span class="k">Printed score</span>
                <span class="v old">{ESC(c['printed'])}</span></span>
            </div>
            <div class="row">{score_cell(c)}</div>
            <div class="row verdict-row">{badge}{note}</div>
          </div>
        </div>"""

    def section(title, n, items):
        body = "\n".join(card_html(c) for c in items) if items else \
            "<p class='empty'>none</p>"
        return (f"<h2>{title} <span class='count'>({n})</span></h2>\n"
                f"<div class='grid'>{body}</div>")

    def summary_stat(label, n, cls):
        return (f'<div class="stat {cls}"><div class="n">{n}</div>'
                f'<div class="l">{label}</div></div>')

    n_true = len(trues)
    n_false = len(falses)
    # printed numeric midpoint per group (old model) for a rough split read
    def avg_printed(items):
        vals = []
        for c in items:
            mm = re.search(r"([01]\.\d{1,2})", c["printed"])
            if mm:
                vals.append(float(mm.group(1)))
        return (f"{sum(vals)/len(vals):.2f}" if vals else "n/a")

    v4_agree = sum(
        1 for c in cards
        if c["is_true"] != (float(c["v4_fire"] or 0) >= 0.5
                            or float(c["v4_smoke"] or 0) >= 0.5))
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fire alert review - media/Gmail</title>
<style>
  :root {{ --true:#1a7f37; --false:#cf222e; --bg:#f6f8fa; --card:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ font: 14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:#1f2328; margin:0; padding:20px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#57606a; margin-bottom:16px; }}
  .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
  .stat {{ background:var(--card); border:1px solid #d0d7de; border-radius:10px;
          padding:10px 16px; min-width:120px; text-align:center; }}
  .stat .n {{ font-size:26px; font-weight:700; }}
  .stat.true .n {{ color:var(--true); }} .stat.false .n {{ color:var(--false); }}
  .stat .l {{ font-size:11px; color:#57606a; }}
  h2 {{ font-size:16px; margin:22px 0 8px; }}
  .count {{ font-weight:400; color:#57606a; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
          gap:14px; }}
  .card {{ background:var(--card); border:1px solid #d0d7de; border-radius:12px;
          overflow:hidden; }}
  .card.true {{ border-left:5px solid var(--true); }}
  .card.false {{ border-left:5px solid var(--false); }}
  .imgwrap {{ background:#000; text-align:center; }}
  .imgwrap img {{ max-width:100%; display:block; margin:0 auto; max-height:220px; }}
  .meta {{ padding:10px 12px 12px; }}
  .file {{ font-size:11px; color:#57606a; margin-bottom:6px; word-break:break-all; }}
  .row {{ display:flex; flex-wrap:wrap; gap:10px 14px; margin-top:6px; }}
  .kv {{ display:flex; flex-direction:column; }}
  .k {{ font-size:10px; text-transform:uppercase; letter-spacing:.04em; color:#6e7781; }}
  .v {{ font-size:15px; font-weight:600; }}
  .v.old {{ color:#b35900; }}
  .verdict-row {{ align-items:center; }}
  .badge {{ font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px; }}
  .badge.true {{ background:#dafbe1; color:var(--true); }}
  .badge.false {{ background:#ffebe9; color:var(--false); }}
  .note {{ font-size:12px; color:#4b5563; margin-left:4px; }}
  .break {{ width:100%; }}
  .tag {{ font-size:11px; padding:2px 8px; border-radius:999px; background:#eff1f3; }}
  .tag.fire {{ background:#fff1e0; color:#b35900; }}
  .tag.smoke {{ background:#eef1ff; color:#4f5bd5; }}
  .tag.both {{ background:#ffe9e9; color:#b00020; }}
  .empty {{ color:#57606a; font-style:italic; }}
  footer {{ margin-top:22px; color:#57606a; font-size:12px; }}
</style>
</head>
<body>
  <h1>&#128293; Fire-alert review — media/Gmail</h1>
  <div class="sub">Telegram fire alerts (sent by the earlier firewatch checkpoint,
  2026-09-06/07) with the printed score on the image, the ACTIVE model (v4) score
  on the CCTV frame, and your TRUE/FALSE verdict. Images are the cropped CCTV
  frames; click an image to open the original screenshot.</div>

  <div class="summary">
    {summary_stat("Total alerts", len(cards), "")}
    {summary_stat("TRUE fires", n_true, "true")}
    {summary_stat("FALSE positives", n_false, "false")}
    {summary_stat("Avg printed score - TRUE", avg_printed(trues), "true")}
    {summary_stat("Avg printed score - FALSE", avg_printed(falses), "false")}
  </div>

  {section("TRUE fires", n_true, trues)}
  {section("FALSE positives", n_false, falses)}

  <footer>Verdicts: {ESC(csv_path)} · V4 threshold gate 0.50 (fire) — a row shows
  &ldquo;V4 would flag&rdquo; when V4&rsquo;s fire <i>or</i> smoke reaches that gate
  on the frame.</footer>
</body>
</html>
"""
    out_path = os.path.join(args.src, OUT)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Wrote {out_path} ({len(cards)} alerts; "
          f"{n_true} TRUE, {n_false} FALSE)")


if __name__ == "__main__":
    main()
