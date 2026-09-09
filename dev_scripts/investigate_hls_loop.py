#!/usr/bin/env python3
"""investigate_hls_loop.py - Selenium driver for the portal Live HLS state loop.

Drives the REAL portal (default https://live.mazr3a.garden) with headless Chrome,
logs in as a portal user, and for each target camera records the Live view's
HLS state machine over time (see portal/static/app.js):

    Connecting <cam>... / HLS starting... / Live (HLS) / HLS starting - waiting
    for video... / Upgrading to live HLS...  (the loop under investigation)

It samples the DOM + <video> decode signals every 500 ms, captures console +
performance (network) logs, and takes screenshots at status transitions.

Usage:
  PORTAL_USER=admin PORTAL_PASS='...' \
    .venv/bin/python dev_scripts/investigate_hls_loop.py \
      [--base https://live.mazr3a.garden] [--cams cam09,cam10] \
      [--secs 60] [--out /tmp/mz_hls_inv]

Only ever reads from the portal. Intended for reproduction/investigation only.
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
USER = os.environ.get("PORTAL_USER", "")
PASS = os.environ.get("PORTAL_PASS", "")

SAMPLE_MS = 0.5
SHOT_EVERY_S = 10.0

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
  return {
    status: st ? st.textContent : null,
    cam: tag ? tag.textContent : null,
    vHidden: hc(v), imgHidden: hc(img), spHidden: hc(sp), ovHidden: hc(ov),
    ready: v ? v.readyState : null,
    paused: v ? v.paused : null,
    ct: v ? +v.currentTime.toFixed(3) : null,
    vw: v ? v.videoWidth : null, vh: v ? v.videoHeight : null,
    frames: q ? q.totalVideoFrames : null,
    decFrames: (v && v.webkitDecodedFrameCount != null) ? v.webkitDecodedFrameCount : null,
    px: px,
    bufLen: bufLen, bufEnd: bufEnd,
    err: (v && v.error) ? v.error.code + ':' + v.error.message : null
  };
})()
"""


def make_driver(shot_dir, headed=False):
    o = Options()
    if headed:
        o.add_argument("--start-maximized")
    else:
        o.add_argument("--headless=new")
        o.add_argument("--window-size=1360,960")
        o.add_argument("--disable-gpu")
        o.add_argument("--use-gl=swiftshader")
        o.add_argument("--enable-unsafe-swiftshader")
    o.add_argument("--no-sandbox")
    o.add_argument("--disable-dev-shm-usage")
    o.add_argument("--autoplay-policy=no-user-gesture-required")
    o.set_capability("goog:loggingPrefs", {"browser": "ALL", "performance": "ALL"})
    d = webdriver.Chrome(options=o)
    d.set_page_load_timeout(60)
    return d


def _cls_has(el, name):
    return name in (el.get_attribute("class") or "").split()


def login(d):
    d.get(PORTAL_BASE + "/")

    def shell_ready():
        app = d.find_element(By.ID, "app")
        lv = d.find_element(By.ID, "login-view")
        return (not _cls_has(app, "hidden")) or (not _cls_has(lv, "hidden"))

    WebDriverWait(d, 40).until(lambda x: shell_ready())
    if not _cls_has(d.find_element(By.ID, "app"), "hidden"):
        return  # already authenticated (session cookie)
    lv = d.find_element(By.ID, "login-view")
    WebDriverWait(d, 15).until(lambda x: not _cls_has(lv, "hidden"))
    f = d.find_element(By.ID, "login-form")
    f.find_element(By.NAME, "username").send_keys(USER)
    f.find_element(By.NAME, "password").send_keys(PASS)
    f.find_element(By.CSS_SELECTOR, "button[type=submit]").click()
    WebDriverWait(d, 40).until(
        lambda x: not _cls_has(d.find_element(By.ID, "app"), "hidden"))
    # stay on Live view (boot defaults there); wait for the camera strip
    if not _cls_has(d.find_element(By.ID, "view-live"), "hidden"):
        d.execute_script("location.hash = '#/live';")
    WebDriverWait(d, 45).until(
        lambda x: x.find_elements(By.CSS_SELECTOR, "#cam-strip .cam-item"))
    time.sleep(0.5)


def keep_alive(d):
    """Synthesize activity so the portal idle-watch doesn't stop the stream."""
    d.execute_script(
        "document.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true}));"
        "document.dispatchEvent(new Event('scroll', {bubbles:true}));")


def sample(d):
    return json.loads(d.execute_script("return JSON.stringify(" + JS_SAMPLE + ")"))


def select_cam(d, cam):
    wait = WebDriverWait(d, 45)
    sel_strip = '#cam-strip .cam-item[data-cam="%s"]' % cam
    try:
        cell = wait.until(
            lambda x: x.find_element(By.CSS_SELECTOR, sel_strip))
    except Exception:
        picker = wait.until(lambda x: x.find_element(By.ID, "cam-picker"))
        d.execute_script("arguments[0].click();", picker)
        sel_grid = '#cam-grid .cam-item[data-cam="%s"]' % cam
        cell = wait.until(lambda x: x.find_element(By.CSS_SELECTOR, sel_grid))
    d.execute_script(
        "arguments[0].scrollIntoView({block:'center'});"
        "arguments[0].click();", cell)


def run_camera(d, cam, secs, outdir, last_status_holder):
    events = []
    shots = []
    last_status = last_status_holder[0]
    t0 = time.time()
    last_shot = 0.0
    last_alive = 0.0
    select_cam(d, cam)
    while time.time() - t0 < secs:
        try:
            s = sample(d)
        except Exception:
            time.sleep(0.5)
            continue
        st = s["status"]
        row = {"cam": cam, "elapsed": round(time.time() - t0, 2), **s}
        events.append(row)
        if st != last_status:
            px = s.get("px") or {}
            print("[%s] %.1fs status -> %r (vH=%s iH=%s frames=%s dec=%s "
                  "ct=%s px=%s)"
                  % (cam, row["elapsed"], st, s["vHidden"], s["imgHidden"],
                     s["frames"], s.get("decFrames"), s["ct"], px.get("sd")))
            last_status = st
            last_status_holder[0] = st
            fname = "%s/%s_t%05.0f_%s.png" % (
                outdir, cam, (time.time() - t0) * 10,
                (st or "?")[:20].replace(" ", "_").replace("/", "_"))
            try:
                d.save_screenshot(fname)
                shots.append(fname)
            except Exception:
                pass
        now = time.time()
        if now - last_alive > 5:
            keep_alive(d)
            last_alive = now
        if now - last_shot > SHOT_EVERY_S:
            try:
                fname = "%s/%s_periodic_%06.0f.png" % (
                    outdir, cam, (now - t0) * 10)
                d.save_screenshot(fname)
                shots.append(fname)
            except Exception:
                pass
            last_shot = now
        time.sleep(SAMPLE_MS)
    # write per-camera artifacts
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
    return events, shots


def summarize(events, cam):
    seq = []
    for e in events:
        if not seq or seq[-1]["status"] != e["status"]:
            seq.append(e)
    lives = [e for e in events if e["status"] == "Live (HLS)"]
    waits = [e for e in events if e["status"]
             and "waiting for video" in e["status"]]
    painted = [e for e in lives if e["frames"] and e["frames"] > 0]
    print("== %s: transitions=%d live_lbl=%d live_with_frames=%d waits=%d"
          % (cam, len(seq), len(lives), len(painted), len(waits)))
    for e in seq:
        print("   %6.1fs  %-40s vH=%s iH=%s frames=%s ct=%s ready=%s paused=%s"
              % (e["elapsed"], (e["status"] or "")[:40], e["vHidden"],
                 e["imgHidden"], e["frames"], e["ct"], e["ready"], e["paused"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=PORTAL_BASE)
    ap.add_argument("--cams", default="cam09,cam10")
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--out", default="/tmp/mz_hls_inv")
    ap.add_argument("--headed", action="store_true",
                    help="run a visible browser on DISPLAY (default: headless)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not (USER and PASS):
        sys.exit("set PORTAL_USER and PORTAL_PASS")
    d = make_driver(a.out, headed=a.headed)
    try:
        login(d)
        print("logged in; navigating to Live")
        last_status = [None]
        for cam in [c.strip() for c in a.cams.split(",") if c.strip()]:
            ev, shots = run_camera(d, cam, a.secs, a.out, last_status)
            summarize(ev, cam)
            print("   screenshots:", len(shots))
    finally:
        d.quit()


if __name__ == "__main__":
    main()
