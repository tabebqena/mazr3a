#!/usr/bin/env python3
"""_diag_hls_browser.py - Selenium driver reproducing the portal Live HLS
state loop (quiet vs motion cameras), with optional mobile emulation + CDP
network throttling to approximate the slower mobile path.

Logs in via a pre-minted portal_session cookie (PORTAL_COOKIE env) so no
plaintext portal password is needed. Samples #live-status, <video> decode
signals, hls.js internals (curHls when reachable) and cumulative HLS network
requests every 0.5 s; captures browser console + performance logs.

Usage:
  PORTAL_COOKIE="$(cat /tmp/mz_portal_cookie.txt)" \
    .venv/bin/python dev_scripts/_diag_hls_browser.py \
      --cams cam01,cam02,cam04 --secs 40 [--mobile] [--latency 220] [--down 1400] \
      [--out /tmp/mz_hls_browser]
"""
import argparse
import csv
import json
import os
import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

PORTAL_BASE = os.environ.get("PORTAL_BASE", "https://live.mazr3a.garden")
COOKIE = os.environ.get("PORTAL_COOKIE", "")
SAMPLE_MS = 0.5

JS_SAMPLE = """
(() => {
  const gid = (i) => document.getElementById(i);
  const v = gid('live-video'), img = gid('live-img'), sp = gid('live-spinner');
  const st = gid('live-status'), tag = gid('live-cam-tag'), ov = gid('live-overlay');
  const q = (v && v.getVideoPlaybackQuality) ? v.getVideoPlaybackQuality() : null;
  let bufEnd = null, bufLen = 0;
  if (v && v.buffered && v.buffered.length) {
    bufLen = v.buffered.length;
    bufEnd = +v.buffered.end(v.buffered.length - 1).toFixed(3);
  }
  const hc = (e) => e ? e.classList.contains('hidden') : true;
  let px = null;
  if (v && v.videoWidth > 0) {
    try {
      const cv = document.createElement('canvas');
      cv.width = 48; cv.height = 27;
      const cx = cv.getContext('2d', { willReadFrequently: true });
      cx.drawImage(v, 0, 0, cv.width, cv.height);
      const dd = cx.getImageData(0, 0, cv.width, cv.height).data;
      let s = 0, nn = cv.width * cv.height;
      for (let i = 0; i < dd.length; i += 4) s += (dd[i] + dd[i + 1] + dd[i + 2]) / 3;
      const mean = s / nn; let varr = 0;
      for (let i = 0; i < dd.length; i += 4) {
        const l = (dd[i] + dd[i + 1] + dd[i + 2]) / 3;
        varr += (l - mean) * (l - mean);
      }
      px = { mean: Math.round(mean), sd: Math.round(Math.sqrt(varr / nn)) };
    } catch (e) { px = { err: String(e) }; }
  }
  // Cumulative HLS network requests (playlists + segment.ts) via resource timing.
  let net = { hls: 0, seg: 0, pl: 0, lastSegN: null, err500: 0 };
  try {
    const res = performance.getEntriesByType('resource');
    for (const r of res) {
      const u = r.name;
      if (u.indexOf('/api/live/') < 0) continue;
      if (u.indexOf('.m3u8') >= 0) { net.pl++; }
      else if (u.indexOf('segment.ts') >= 0) {
        net.seg++;
        net.hls++;
        const m = u.match(/[?&]n=(\\d+)/);
        if (m) net.lastSegN = +m[1];
        if (r.responseStatus >= 400) net.err500++;
      } else if (u.indexOf('/hls/') >= 0) { net.hls++; }
    }
  } catch (e) {}
  // hls.js instance internals (top-level let in app.js -> visible to the console)
  let h = null;
  try {
    if (typeof curHls !== 'undefined' && curHls) {
      const cfg = curHls.config || {};
      h = {
        level: curHls.currentLevel, manualLevel: curHls.manualLevel,
        nextLoad: curHls.nextLoadLevel, nudgeMax: cfg.nudgeMaxRetry,
        liveSync: cfg.liveSyncDurationCount
      };
    }
  } catch (e) {}
  let url = '';
  try { url = v ? (v.currentSrc || v.src || '') : ''; } catch (e) {}
  return {
    status: st ? st.textContent : null,
    cam: tag ? tag.textContent : null,
    vHidden: hc(v), imgHidden: hc(img), spHidden: hc(sp),
    ready: v ? v.readyState : null,
    paused: v ? v.paused : null,
    ct: v ? +v.currentTime.toFixed(3) : null,
    vw: v ? v.videoWidth : null, vh: v ? v.videoHeight : null,
    frames: q ? q.totalVideoFrames : null,
    decFrames: (v && v.webkitDecodedFrameCount != null) ? v.webkitDecodedFrameCount : null,
    px: px, bufLen: bufLen, bufEnd: bufEnd,
    err: (v && v.error) ? v.error.code + ':' + v.error.message : null,
    net: net, hls: h, src: url.slice(0, 120)
  };
})()
"""


def make_driver(outdir, mobile=False):
    o = Options()
    o.add_argument("--headless=new")
    o.add_argument("--disable-gpu")
    o.add_argument("--use-gl=swiftshader")
    o.add_argument("--enable-unsafe-swiftshader")
    o.add_argument("--no-sandbox")
    o.add_argument("--disable-dev-shm-usage")
    o.add_argument("--autoplay-policy=no-user-gesture-required")
    o.add_argument("--window-size=1360,960")
    o.set_capability("goog:loggingPrefs", {"browser": "ALL", "performance": "ALL"})
    d = webdriver.Chrome(options=o)
    d.set_page_load_timeout(60)
    if mobile:
        d.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
            "width": 390, "height": 844, "deviceScaleFactor": 3,
            "mobile": True, "screenWidth": 390, "screenHeight": 844})
        d.execute_cdp_cmd("Emulation.setUserAgentOverride", {
            "userAgent": ("Mozilla/5.0 (Linux; Android 13; Pixel 7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/153.0 Mobile Safari/537.36")})
    return d


def throttle(d, latency_ms, down_kbps):
    d.execute_cdp_cmd("Network.enable", {})
    d.execute_cdp_cmd("Network.emulateNetworkConditions", {
        "offline": False,
        "latency": latency_ms,
        "downloadThroughput": int(down_kbps) * 1024,
        "uploadThroughput": 2 * 1024 * 1024,
        "connectionType": "cellular4g"})


def _cls_has(el, name):
    return name in (el.get_attribute("class") or "").split()


def inject_hls_tune(d, tune_spec):
    """Wrap window.Hls with a Proxy that merges override keys into any Hls()
    config, to test tuning WITHOUT editing/deploying app.js. tune_spec is a
    ';'-separated list of 'key:value' (numeric values supported)."""
    if not tune_spec:
        return
    kv = []
    for pair in tune_spec.split(";"):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        k = k.strip()
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            v = v.strip()
        kv.append("%s: %r" % (k, v))
    js = ("(() => { if (window.__hlsTuned) return; window.__hlsTuned = true;"
          " const R = window.Hls; if (!R) return;"
          " const EXTRA = { %s };"
          " window.Hls = new Proxy(R, { construct(t, a) {"
          "  const cfg = Object.assign({}, a[0] || {});"
          "  for (const k in EXTRA) cfg[k] = EXTRA[k];"
          "  return Reflect.construct(t, [cfg]); } });"
          "})();" % ", ".join(kv))
    d.execute_script(js)
    print("hls.js tuning injected:", tune_spec)


def login_cookie(d):
    # 1) land once on the origin (so CDP setCookie has a document context),
    # 2) inject the session cookie via CDP (same-domain, https),
    # 3) d.refresh() = REAL reload so boot() re-runs authenticated
    #    (a fragment-only d.get would be a same-document nav and NOT reload).
    d.get(PORTAL_BASE + "/")
    d.execute_cdp_cmd("Network.enable", {})
    ok = d.execute_cdp_cmd("Network.setCookie", {
        "name": "portal_session", "value": COOKIE,
        "url": PORTAL_BASE + "/", "path": "/",
        "secure": True, "httpOnly": True, "sameSite": "Lax"})
    print("setCookie ok:", ok)
    d.refresh()
    WebDriverWait(d, 45).until(
        lambda x: not _cls_has(x.find_element(By.ID, "app"), "hidden"))
    WebDriverWait(d, 45).until(
        lambda x: x.find_elements(By.CSS_SELECTOR, "#cam-strip .cam-item"))
    time.sleep(1.0)


def keep_alive(d):
    d.execute_script(
        "document.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true}));"
        "document.dispatchEvent(new Event('scroll', {bubbles:true}));")


def sample(d):
    return json.loads(d.execute_script("return JSON.stringify(" + JS_SAMPLE + ")"))


def select_cam(d, cam):
    wait = WebDriverWait(d, 45)
    sel_strip = '#cam-strip .cam-item[data-cam="%s"]' % cam
    try:
        cell = wait.until(lambda x: x.find_element(By.CSS_SELECTOR, sel_strip))
    except Exception:
        picker = wait.until(lambda x: x.find_element(By.ID, "cam-picker"))
        d.execute_script("arguments[0].click();", picker)
        sel_grid = '#cam-grid .cam-item[data-cam="%s"]' % cam
        cell = wait.until(lambda x: x.find_element(By.CSS_SELECTOR, sel_grid))
    d.execute_script(
        "arguments[0].scrollIntoView({block:'center'}); arguments[0].click();",
        cell)


def run_camera(d, cam, secs, outdir):
    events = []
    last_status = None
    t0 = time.time()
    last_alive = 0.0
    select_cam(d, cam)
    while time.time() - t0 < secs:
        try:
            s = sample(d)
        except Exception:
            time.sleep(0.3)
            continue
        st = s["status"]
        row = {"cam": cam, "elapsed": round(time.time() - t0, 2), **s}
        events.append(row)
        if st != last_status:
            px = s.get("px") or {}
            net = s.get("net") or {}
            h = s.get("hls")
            print("[%s] %6.1fs status -> %-34r frames=%s dec=%s ct=%s px_sd=%s "
                  "buf=%s netSeg=%s(lastN=%s) hls=%s"
                  % (cam, row["elapsed"], st, s["frames"], s.get("decFrames"),
                     s["ct"], px.get("sd"), s.get("bufEnd"),
                     net.get("seg"), net.get("lastSegN"), h))
            last_status = st
        now = time.time()
        if now - last_alive > 5:
            keep_alive(d)
            last_alive = now
        time.sleep(SAMPLE_MS)
    with open("%s/%s_events.csv" % (outdir, cam), "w", newline="") as fh:
        if events:
            w = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
            w.writeheader()
            w.writerows(events)
    for lg, nm in (("browser", "console"), ("performance", "perf")):
        try:
            lines = [e.get("message", "") for e in d.get_log(lg)]
        except Exception as ex:
            lines = ["(log unavail: %s)" % ex]
        with open("%s/%s_%s.log" % (outdir, cam, nm), "w") as fh:
            fh.write("\n".join(lines))
    return events


def summarize(events, cam):
    seq = []
    for e in events:
        if not seq or seq[-1]["status"] != e["status"]:
            seq.append(e)
    lives = [e for e in events if e["status"] in ("Live (HLS)", "Live")]
    waits = [e for e in events if e["status"] and "waiting for video" in e["status"]]
    painted = [e for e in lives if e.get("frames") and e["frames"] > 0]
    first_live = None
    for e in lives:
        if e.get("frames") and e["frames"] > 0:
            first_live = e["elapsed"]
            break
    net_max = max((e.get("net", {}).get("seg") or 0) for e in events) if events else 0
    print("== %s: transitions=%d live_lbl=%d live_with_frames=%d waits=%d "
          "first_live_with_frames=%.1fs max_seg_fetches=%d"
          % (cam, len(seq), len(lives), len(painted), len(waits),
             first_live or -1, net_max))
    for e in seq:
        px = e.get("px") or {}
        print("   %6.1fs  %-40s vH=%s frames=%s ct=%s ready=%s px_sd=%s netSeg=%s lastN=%s"
              % (e["elapsed"], (e["status"] or "")[:40], e["vHidden"],
                 e["frames"], e["ct"], e["ready"], px.get("sd"),
                 e.get("net", {}).get("seg"), e.get("net", {}).get("lastSegN")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=PORTAL_BASE)
    ap.add_argument("--cams", default="cam01,cam02,cam04")
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--out", default="/tmp/mz_hls_browser")
    ap.add_argument("--mobile", action="store_true")
    ap.add_argument("--latency", type=int, default=0, help="extra RTT ms (CDP)")
    ap.add_argument("--down", type=int, default=0, help="download cap kbps (0=off)")
    a = ap.parse_args()
    if not COOKIE:
        sys.exit("set PORTAL_COOKIE (minted session cookie)")
    os.makedirs(a.out, exist_ok=True)
    d = make_driver(a.out, mobile=a.mobile)
    try:
        if a.latency or a.down:
            throttle(d, a.latency, a.down)
            print("network throttle: latency=%sms down=%skbps" % (a.latency, a.down))
        login_cookie(d)
        print("logged in (cookie); mobile=%s" % a.mobile)
        for cam in [c.strip() for c in a.cams.split(",") if c.strip()]:
            events = run_camera(d, cam, a.secs, a.out)
            summarize(events, cam)
    finally:
        d.quit()


if __name__ == "__main__":
    main()
