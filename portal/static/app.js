/* mazr3a CCTV portal SPA - vanilla JS.
   Views: Login, Live (single camera; HLS via the portal's go2rtc proxy with
   hls.js and a detect-snapshot fallback, idle time watch), Frigate Events &
   detections, Firewatch fire detections & alerts. Talks to the portal /api/*
   (same origin, session cookie). */
'use strict';

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

const state = {
  me: null,
  settings: null,
  cameras: [],
  live: { cam: null, playing: false, mode: '', sound: false, imgLive: false,
          revealed: false },   // true once this HLS attempt dropped the poster
  // imgLive: the <img> is the poster shown while an HLS attempt is starting
  //          (or the frozen last frame while an online camera auto-retries).
  // mode: '' | 'hls' | 'offline'
  idleSec: 30,
};

const LIVE_POLL_MS = 1100;   // snapshot fallback rate (~1 fps, detect fps)
const SNAPSHOT_RETRY_MS = 15000;  // auto-retry interval: snapshot -> live HLS
const OFFLINE_CHECK_MS = 8000;    // how often to re-check an offline camera
const HLS_PAINT_WAIT_MS = 3500;   // post-reveal: claim Live only once a real
                                  // video frame has been decoded (frame-based)
// Retry when HLS buffered but no video frame ever rendered (blackRevert path).
const HLS_VIDEO_RETRY_MS = 6000;
// Post-"Live" watchdog: go2rtc HLS is a tiny ~2 x 0.5 s sliding live window and
// can stall right after a frame renders over a slow path, freezing a black
// "Live (HLS)". If no NEW video frame decodes for this long while Live is
// claimed, restart HLS (fresh session) instead of sitting on a black/frozen.
const HLS_STALL_WATCH_MS = 4000;
// True when this browser can play HLS at all (hls.js MSE or native Safari HLS).
const HLS_SUPPORTED = !!(window.Hls && window.Hls.isSupported())
  || (function () {
       try {
         const v = document.createElement('video');
         return !!(v.canPlayType && v.canPlayType('application/vnd.apple.mpegurl'));
       } catch (e) { return false; }
     })();
let liveTimer = null;
let idleTimer = null;
let liveTok = 0;             // guards stale async callbacks after switch/stop
let curHls = null;           // active hls.js instance (destroy on stop/switch)
let frameWatchTimer = null;  // watchdog: give up waiting for the first media
let retryTimer = null;       // auto-retry: snapshot fallback -> HLS live
let offlineTimer = null;     // periodic re-check while a camera is offline
let paintCheck = null;       // post-reveal: confirm frames really render
let stallWatch = null;       // post-Live: restart if no new frame (black/frozen)

let fwDets = {};             // fire frame id -> detections (for box overlay)

/* ---------------- helpers ---------------- */
function esc(s) {
  const map = {
    '&': '\u0026amp;', '<': '\u0026lt;', '>': '\u0026gt;',
    '"': '\u0026quot;', "'": '\u0026#39;',
  };
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => map[c]);
}
function fmtDT(sec) {
  if (!sec) return '—';
  return new Date(sec * 1000).toLocaleString();
}
function toast(msg) {
  const t = $('#global-err');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add('hidden'), 5000);
}
async function api(path, opts) {
  const init = Object.assign({ credentials: 'same-origin' }, opts || {});
  if (init.body && typeof init.body !== 'string') init.body = JSON.stringify(init.body);
  let resp;
  try {
    resp = await fetch(path, init);
  } catch (e) {
    throw new Error('network error');
  }
  if (resp.status === 401) { showLogin(); throw new Error('unauthorized'); }
  if (!resp.ok) {
    let d = {};
    try { d = await resp.json(); } catch (e) { /* ignore */ }
    throw new Error(d.detail || ('HTTP ' + resp.status));
  }
  return resp.json();
}

/* ---------------- login / app shell ---------------- */
function showLogin() {
  $('#app').classList.add('hidden');
  $('#login-view').classList.remove('hidden');
}
function hideLogin() {
  $('#login-view').classList.add('hidden');
  $('#app').classList.remove('hidden');
}

$('#login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const err = $('#login-error');
  err.classList.add('hidden');
  const fd = new FormData(e.target);
  try {
    await api('/api/login', {
      method: 'POST',
      body: JSON.stringify({ username: fd.get('username'), password: fd.get('password') }),
    });
    boot();
  } catch (ex) {
    err.textContent = ex.message === 'unauthorized' ? 'Invalid username or password' : ex.message;
    err.classList.remove('hidden');
  }
});

$('#logout-btn').addEventListener('click', async () => {
  stopStream();
  try { await api('/api/logout', { method: 'POST' }); } catch (e) { /* ignore */ }
  showLogin();
});

/* ---------------- boot / router ---------------- */
async function boot() {
  try {
    state.me = await api('/api/me');
  } catch (e) { return; }
  hideLogin();
  $('#user-chip').textContent = state.me.username;
  state.settings = await api('/api/settings');
  // Idle stop is set by an admin in portal.conf (STREAM_IDLE_TIMEOUT_S); there is
  // no in-UI control, so every user gets the server-configured value.
  state.idleSec = Math.max(15, parseInt(state.settings.stream_idle_timeout_s, 10) || 30);
  // Bottom-most version label: server APP_VERSION (the HTML text is the fallback).
  const vn = $('#ver-no');
  if (vn && state.settings.app_version) vn.textContent = state.settings.app_version;
  await loadCameras();
  window.addEventListener('hashchange', onRoute);
  onRoute();
}

function onRoute() {
  const raw = (location.hash || '#/live').replace(/^#\//, '');
  const view = ['live', 'events', 'fire'].indexOf(raw) >= 0 ? raw : 'live';
  // Live is full-bleed (fills the screen, no dead scroll); other views scroll.
  document.body.classList.toggle('live-full', view === 'live');
  if (view !== 'live') stopStream();           // only the Live view streams
  $$('#nav a').forEach(a => a.classList.toggle('active', a.dataset.view === view));
  ['live', 'events', 'fire'].forEach(v =>
    $('#view-' + v).classList.toggle('hidden', v !== view));
  if (view === 'live') ensureLive();
  else if (view === 'events') loadEvents();
  else if (view === 'fire') reloadFire();   // page-based: reset to page 1 on entry
}

async function loadCameras() {
  const data = await api('/api/cameras');
  state.cameras = data.cameras || [];
  refreshCamSelects();                 // Events/Fire selects keep a dropdown

  // Reopen the LAST camera the user watched (persisted), else the default.
  let def = null;
  try { def = localStorage.getItem('portal.lastCam'); } catch (e) { /* ignore */ }
  if (!def || camNames().indexOf(def) < 0) {
    def = state.me.default_camera || data.default_camera
      || (state.cameras.find(c => c.enabled) || {}).name || camNames()[0];
  }
  state.live.cam = camNames().indexOf(def) >= 0 ? def : camNames()[0] || null;
  renderCamPicker();                   // thumbnail row + modal grid + labels
}

/* Fill the Events/Fire camera <select>s from state.cameras. The Live view no
   longer uses a <select> - cameras are picked from the thumbnail row/modal. */
function refreshCamSelects() {
  const opts = camNames().map(n => {
    const c = state.cameras.find(x => x.name === n);
    const tag = c && !c.online ? ' (offline)' : '';
    return '<option value="' + esc(n) + '">' + esc(n) + tag + '</option>';
  }).join('');
  $('#ev-cam').innerHTML = '<option value="">all cameras</option>' + opts;
  $('#fw-cam').innerHTML = '<option value="">all cameras</option>' + opts;
}

/* ---------------- Live view: HLS (go2rtc) via hls.js ------------------------ */
// go2rtc in this Frigate 0.17.2 build has no working MSE-over-WS and no HTTP
// MSE endpoint; its tunnel-friendly live transport is HLS (master + ~0.5 s .ts
// segments), proxied SAME-ORIGIN through the portal at
//   /api/live/<cam>/hls/stream.m3u8?src=<cam>
// hls.js plays it (MSE inside the page); Safari plays it natively. A poster
// (the latest detect frame) shows while HLS starts. If HLS cannot start the
// camera is classified from Frigate's online flag: OFFLINE -> a clean offline
// state (no spinner, no snapshot feed); ONLINE -> HLS startup latency, so we
// keep the frozen poster and auto-retry HLS in the background.

function hlsFallback(cam, tok) {
  /* HLS could not start. Stop the attempt, then classify WHY:
     - camera OFFLINE (Frigate online flag false) -> a clean "offline" state,
       no spinner and NO ~1 fps snapshot feed;
     - camera ONLINE but slow to start -> that is LATENCY, so we keep the last
       frozen poster (no snapshot feed) and auto-retry HLS in the background.
     Frigate's online flag (camera_fps) is authoritative, so HLS startup
     latency is never mistaken for an offline camera. */
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || state.live.mode !== 'hls') return;
  clearFrameWatch();                 // no stale first-media watchdog
  clearPaintCheck();
  clearStallWatch();
  if (curHls) { try { curHls.destroy(); } catch (e) { /* ignore */ } curHls = null; }
  state.live.imgLive = false;        // freeze the poster; stop the poster poll
  if (liveTimer) { clearTimeout(liveTimer); liveTimer = null; }
  const video = $('#live-video');
  try { video.pause(); video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; }
  $('#live-video').classList.add('hidden');
  $('#live-sound').classList.add('hidden');
  hideSpinner();
  decideAfterHlsFail(cam, tok);
}

async function decideAfterHlsFail(cam, tok) {
  await fetchCameras();              // fresh Frigate online flags
  if (!state.live.playing || cam !== state.live.cam
      || state.live.mode !== 'hls') return;
  if (!isOnline(cam)) { enterOffline(cam); return; }
  // Camera is ONLINE -> just latency/startup: keep the frozen poster visible,
  // no spinner, no snapshot feed, and auto-retry HLS in the background.
  $('#live-img').classList.remove('hidden');
  $('#live-status').textContent = 'HLS starting - auto-retrying\u2026';
  scheduleLiveRetry(cam);
}

/* Refresh the camera list / online flags from the portal (/api/cameras).
   Re-renders the thumbnail row/modal only when the list really changed
   (names/enabled/online) so frequent re-checks do not reload thumbnails. */
async function fetchCameras() {
  try {
    const data = await api('/api/cameras');
    const list = data.cameras || [];
    if (list.length) {
      state.cameras = list;
      if (camsChanged(list)) {
        refreshCamSelects();
        renderCamPicker();             // fresh online flags + thumbnails
      }
    }
    return data;
  } catch (e) { return null; }
}
let camsSig = '';   // last-rendered camera signature (see camsChanged)
function camsChanged(list) {
  const sig = JSON.stringify(list.map(c => [c.name, c.enabled, !!c.online]));
  if (sig === camsSig) return false;
  camsSig = sig;
  return true;
}
/* Frigate online flag for a camera; unknown -> assume ONLINE so HLS startup
   latency is never reported as an offline camera. */
function isOnline(cam) {
  const c = state.cameras.find(x => x.name === cam);
  return c ? !!c.online : true;
}

/* Camera is offline: no spinner, no snapshot feed - just a clear offline
   overlay. Auto re-checks periodically and starts live when the camera is
   back online. */
function enterOffline(cam) {
  resetZoom();                 // clear any zoom/pan from the live frame
  state.live.mode = 'offline';
  state.live.imgLive = false;
  state.live.playing = false;        // idle/activity semantics like "stopped"
  liveTok++;
  if (liveTimer) { clearTimeout(liveTimer); liveTimer = null; }
  if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
  clearFrameWatch();
  if (curHls) { try { curHls.destroy(); } catch (e) { /* ignore */ } curHls = null; }
  const video = $('#live-video');
  try { video.pause(); video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; }
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.add('hidden');
  $('#live-sound').classList.add('hidden');
  hideSpinner();
  $('#live-status').textContent = cam + ' is offline';
  showOverlay('Camera ' + cam + ' is offline.');
  if (offlineTimer) { clearInterval(offlineTimer); offlineTimer = null; }
  offlineTimer = setInterval(async () => {
    const data = await fetchCameras();
    if (state.live.mode !== 'offline') {        // view changed/stopped
      clearInterval(offlineTimer); offlineTimer = null;
      return;
    }
    if (data && state.live.cam === cam && isOnline(cam)) {
      clearInterval(offlineTimer); offlineTimer = null;
      startStream(cam);                          // camera is back -> live
    }
  }, OFFLINE_CHECK_MS);
}

/* While an ONLINE camera's HLS is slow/unavailable (latency), keep retrying to
   upgrade to live in the background - the frozen poster stays up. Stops when
   the view stops (idle, leaving Live, switching) or the camera is offline. */
function scheduleLiveRetry(cam, delayMs) {
  if (retryTimer || !state.live.playing || !HLS_SUPPORTED
      || state.live.imgLive) return;
  retryTimer = setTimeout(() => {
    retryTimer = null;
    if (!state.live.playing || state.live.cam !== cam
        || state.live.mode !== 'hls' || state.live.imgLive) return;
    $('#live-status').textContent = 'Upgrading to live HLS\u2026';
    startHls(cam, liveTok);   // background attempt; the frozen poster stays up
  }, delayMs || SNAPSHOT_RETRY_MS);
}

function resetIdle() {
  if (idleTimer) clearTimeout(idleTimer);
  if (!state.live.playing) return;
  idleTimer = setTimeout(onIdleTimeout, Math.max(15, state.idleSec) * 1000);
}
function onIdleTimeout() {
  if (!state.live.playing) return;
  stopStream();
  const s = state.idleSec;
  const txt = s >= 60 ? Math.round(s / 60) + ' minutes' : s + ' seconds';
  showOverlay('Live stopped after ' + txt + ' without activity.');
}
['mousemove', 'mousedown', 'keydown', 'touchstart', 'pointerdown', 'wheel', 'scroll']
  .forEach(ev => document.addEventListener(ev, () => {
    if (state.live.playing) resetIdle();   // any user activity restarts the clock
  }, { passive: true }));

// When the tab is hidden nobody is watching - restart the countdown so a
// backgrounded Live view still stops after the idle window (saves bandwidth).
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.live.playing) resetIdle();
});

function showOverlay(msg) {
  $('#overlay-msg').textContent = msg;
  $('#live-overlay').classList.remove('hidden');
}
function hideOverlay() {
  $('#live-overlay').classList.add('hidden');
}
function showSpinner() {
  const s = $('#live-spinner');
  if (s) s.classList.remove('hidden');
}
function hideSpinner() {
  const s = $('#live-spinner');
  if (s) s.classList.add('hidden');
}
function showVideo() {
  $('#live-video').classList.remove('hidden');
  $('#live-img').classList.add('hidden');
}
/* Live audio: the portal HLS carries AAC (go2rtc <cam>_portal transcode) but
   the <video> must start muted so browsers allow autoplay. Sound is opt-in via
   the #live-sound button; clicking it is a user gesture (always permitted), and
   once a stream is already playing muted the last choice is re-applied safely. */
function applyLiveSound() {
  const video = $('#live-video');
  video.muted = !state.live.sound;
  if (!video.muted) video.volume = 1;
  syncSoundUI();
}
function toggleLiveSound() {
  state.live.sound = !state.live.sound;
  applyLiveSound();
  if (state.live.sound) {
    const p = $('#live-video').play();   // re-engage if a mute stopped it
    if (p && p.catch) p.catch(() => {});
  }
}
function syncSoundUI() {
  const b = $('#live-sound');
  if (!b) return;
  const on = !!state.live.sound;
  b.textContent = on ? '\u{1F50A}' : '\u{1F507}';   // 🔊 / 🔇
  b.title = on ? 'Mute' : 'Enable sound';
  b.setAttribute('aria-label', b.title);
  b.classList.toggle('sound-on', on);
}

function ensureLive() {
  if (!state.live.cam || state.live.playing) return;
  startStream(state.live.cam);
}

function startStream(cam) {
  stopStream();
  const tok = ++liveTok;
  state.live.cam = cam;
  syncCamUI();                        // highlight the active camera in the picker
  if (!isOnline(cam)) { enterOffline(cam); return; }   // offline: no spinner/feed
  state.live.playing = true;
  state.live.mode = 'hls';
  state.live.imgLive = true;      // poster <img> shown while HLS warms up
  hideOverlay();
  showSpinner();                  // poster image + spinner until the first HLS media
  scheduleFrame(cam, tok);        // show the freshest detect frame immediately
  startHls(cam, tok);             // ...while HLS warms up underneath
  resetIdle();
}

/* Reveal the live video once real media has been BUFFERED - NOT on the media
   'playing' event (which fires before any frame exists and caused a black gap
   while go2rtc starts a camera on demand), and NOT on painted-frame detection
   (requestVideoFrameCallback / totalVideoFrames never fire for a <video> that
   is still fully covered by the poster <img>). Data-level signals - hls.js
   FRAG_BUFFERED and the video 'loadeddata' event - fire as soon as real media
   is in the buffer, even while the video sits under the poster. The poster +
   spinner stay up during the true no-data latency window; if no media ever
   arrives the watchdog classifies the camera (offline state vs online retry). */
// Watchdog before classifying why HLS never delivered. It does NOT decide
// offline by itself - decideAfterHlsFail consults Frigate's online flag, so a
// shorter wait never mislabels slow (latency) cameras as offline.
const HLS_FIRST_FRAME_WAIT_MS = 10000;
function armFirstFrameWatch(cam, tok) {
  if (frameWatchTimer) return;                  // already waiting/done
  frameWatchTimer = setTimeout(() => {          // watchdog: no media in time
    frameWatchTimer = null;
    clearFrameWatch();
    if (state.live.playing && tok === liveTok && cam === state.live.cam
        && state.live.mode === 'hls') hlsFallback(cam, tok);
  }, HLS_FIRST_FRAME_WAIT_MS);
}
function firstMediaReady(cam, tok) {   // first real media buffered -> reveal
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || state.live.mode !== 'hls' || state.live.revealed) return;
  firstHlsFrame(cam, tok);
}
function clearFrameWatch() {
  if (frameWatchTimer) { clearTimeout(frameWatchTimer); frameWatchTimer = null; }
}

function startHls(cam, tok) {
  state.live.mode = 'hls';
  state.live.revealed = false;   // each attempt may reveal the poster once
  const video = $('#live-video');
  // Keep the <video> element visible UNDER the poster <img> (CSS z-index) so
  // the browser keeps decoding while we wait; the poster is removed only once
  // real media is buffered (FRAG_BUFFERED / loadeddata -> firstMediaReady).
  video.muted = true;                 // autoplay-safe; sound restored on play
  $('#live-video').classList.remove('hidden');
  $('#live-sound').classList.remove('hidden');
  syncSoundUI();
  // Same-origin HLS through the portal (behind the session cookie):
  // https://<portal>/api/live/<cam>/hls/stream.m3u8?src=<cam>
  // The backend serves the <cam>_portal AAC source.
  const url = '/api/live/' + encodeURIComponent(cam) +
    '/hls/stream.m3u8?src=' + encodeURIComponent(cam);
  $('#live-status').textContent = 'Connecting ' + cam + '…';
  armFirstFrameWatch(cam, tok);       // watchdog: snapshot if no media arrives

  if (window.Hls && Hls.isSupported()) {
    // hls.js -> MediaSource in the page (Chrome/Firefox/Edge etc.)
    const hls = curHls = new Hls({
      enableWorker: true,
      // go2rtc's HLS here is a very shallow live window (~2 x 0.5 s segments).
      // lowLatencyMode pins the player to the live edge; over a high-latency
      // path it stalls re-fetching the same edge segments (black / frozen), so
      // play a little behind the edge with a tolerant, nudge-friendly config.
      lowLatencyMode: false,
      liveSyncDurationCount: 4,     // buffer a few segments behind the live edge
      liveMaxLatencyDurationCount: 10,
      maxBufferLength: 20,
      backBufferLength: 30,
      maxLiveSyncPlaybackRate: 1.5,
      nudgeOffset: 0.5,             // restart playback quickly on a small stall
      nudgeMaxRetry: 10,
    });
    let netErrs = 0;              // give up after repeated network failures
    video.addEventListener('loadeddata', () => firstMediaReady(cam, tok),
                           { once: true });
    hls.loadSource(url);
    hls.attachMedia(video);
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      if (state.live.playing && tok === liveTok && cam === state.live.cam) {
        video.play().catch(() => {});   // muted autoplay
      }
    });
    hls.on(Hls.Events.FRAG_BUFFERED, (e, data) => {
      if (!data || !data.frag) return;
      // Audio-only fragments (AAC) can buffer before any video keyframe - do
      // NOT drop the poster on those (that caused a black "Live (HLS)" on
      // camera switch). Reveal only when video data is in the buffer.
      if (data.frag.type === 'audio' || data.frag.type === 'subtitle') return;
      firstMediaReady(cam, tok);
    });
    hls.on(Hls.Events.ERROR, (e, data) => {
      if (!data || !data.fatal) return;
      if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
        if (++netErrs >= 6) hlsFallback(cam, tok);   // classify: offline vs retry
        else hls.startLoad();
      } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
        hls.recoverMediaError();
      } else {
        hlsFallback(cam, tok);
      }
    });
    return;
  }

  if (video.canPlayType('application/vnd.apple.mpegurl')) {
    // Safari / iOS native HLS
    const onErr = () => hlsFallback(cam, tok);
    video.addEventListener('error', onErr, { once: true });
    video.addEventListener('loadeddata', () => firstMediaReady(cam, tok),
                           { once: true });
    video.src = url;
    video.play().catch(() => hlsFallback(cam, tok));
    return;
  }

  $('#live-status').textContent = 'HLS unsupported in this browser';
  hlsFallback(cam, tok);
}

/* A real video data fragment has buffered -> reveal the <video> (drop the
   poster). "Live (HLS)" is only claimed once a frame actually renders
   (startPaintCheck) - otherwise we keep the poster and auto-retry instead of
   a black screen that says Live. */
function firstHlsFrame(cam, tok) {
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || state.live.mode !== 'hls') return;
  clearFrameWatch();
  clearPaintCheck();
  clearStallWatch();
  state.live.revealed = true;   // reveal only once per attempt
  state.live.imgLive = false;
  if (liveTimer) { clearTimeout(liveTimer); liveTimer = null; }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; }
  showVideo();                        // hide poster, show the <video>
  hideSpinner();
  $('#live-status').textContent = 'HLS starting\u2026';
  startPaintCheck(cam, tok);
}

/* Honest "is a frame really decoding?" counter: totalVideoFrames when the
   browser exposes it (MSE/hls.js), else webkitDecodedFrameCount (Safari).
   Returns -1 when the browser reports neither (can't verify frame decode). */
function videoFrames(video) {
  if (!video) return -1;
  if (video.getVideoPlaybackQuality) {
    const q = video.getVideoPlaybackQuality();
    if (q && typeof q.totalVideoFrames === 'number') return q.totalVideoFrames;
  }
  if (video.webkitDecodedFrameCount !== undefined) {
    return video.webkitDecodedFrameCount;
  }
  return -1;
}

/* After revealing, confirm the <video> is really decoding NEW frames before we
   claim "Live (HLS)". The AAC audio track decodes and advances currentTime even
   when the video track never decodes, so a media-clock check alone would label
   a black feed "Live (HLS)". Only a growing decoded-frame counter is accepted.
   Browsers with no frame counter fall back to the media clock (best effort). */
function startPaintCheck(cam, tok) {
  const video = $('#live-video');
  if (!video) return;
  clearPaintCheck();
  clearStallWatch();
  const canCount = videoFrames(video) !== -1;
  const fr0 = canCount ? videoFrames(video) : 0;
  const t0 = Date.now();
  let lastT = video.currentTime;
  paintCheck = setInterval(() => {
    if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
        || state.live.mode !== 'hls' || state.live.imgLive) {
      clearPaintCheck(); return;
    }
    const fr = canCount ? videoFrames(video) : -1;
    // painted = a new video frame was actually decoded (not just audio).
    const painted = canCount ? fr > fr0 : video.currentTime !== lastT;
    lastT = video.currentTime;
    if (painted && video.readyState >= 2) {
      clearPaintCheck();
      $('#live-status').textContent = 'Live (HLS)';
      applyLiveSound();
      armStallWatch(cam, tok);   // keep it honest: restart if the picture freezes
      return;
    }
    if (Date.now() - t0 > HLS_PAINT_WAIT_MS) {
      clearPaintCheck();
      blackRevert(cam, tok);          // buffered but never decoded a frame
    }
  }, 250);
}
function clearPaintCheck() {
  if (paintCheck) { clearInterval(paintCheck); paintCheck = null; }
}

/* Post-"Live" watchdog. go2rtc HLS serves only a tiny ~2 x 0.5 s sliding live
   window; over a high-latency path hls.js can stall right after a frame
   renders, leaving a frozen/black "Live (HLS)". If no NEW video frame decodes
   for HLS_STALL_WATCH_MS while we claim Live, restart HLS (fresh session)
   instead of sitting on a black/frozen label. Disabled where the browser gives
   no frame counter. */
function armStallWatch(cam, tok) {
  clearStallWatch();
  const video = $('#live-video');
  if (!video || videoFrames(video) < 0) return;   // can't count frames -> skip
  let last = videoFrames(video);
  let since = Date.now();
  stallWatch = setInterval(() => {
    if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
        || state.live.mode !== 'hls') { clearStallWatch(); return; }
    const f = videoFrames(video);
    if (f < 0) { clearStallWatch(); return; }
    if (f > last) { last = f; since = Date.now(); return; }
    if (Date.now() - since > HLS_STALL_WATCH_MS) {
      clearStallWatch();
      blackRevert(cam, tok);   // no new frame for a while -> retry (fresh session)
    }
  }, 500);
}
function clearStallWatch() {
  if (stallWatch) { clearInterval(stallWatch); stallWatch = null; }
}

/* Media was buffered but no video frame ever rendered (black): do NOT claim
   live. Keep the frozen poster, stop this HLS attempt, and retry shortly so
   the view self-recovers instead of showing a black "Live (HLS)". */
function blackRevert(cam, tok) {
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || state.live.mode !== 'hls') return;
  clearPaintCheck();
  clearStallWatch();
  state.live.revealed = true;         // this attempt already revealed once
  state.live.imgLive = false;         // frozen poster (no ~1 fps feed)
  if (curHls) { try { curHls.destroy(); } catch (e) { /* ignore */ } curHls = null; }
  const video = $('#live-video');
  try { video.pause(); video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; }
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.remove('hidden');
  hideSpinner();
  $('#live-status').textContent = 'HLS starting - waiting for video\u2026';
  scheduleLiveRetry(cam, HLS_VIDEO_RETRY_MS);
}

/* Refresh the poster <img> while an HLS attempt is starting (imgLive true).
   Self-schedules until the video reveals or the attempt fails (poster then
   freezes - no persistent ~1 fps snapshot feed). */
function scheduleFrame(cam, tok) {
  if (!state.live.playing || cam !== state.live.cam || tok !== liveTok
      || !state.live.imgLive) return;
  const img = $('#live-img');
  $('#live-img').classList.remove('hidden');  // poster above the warming video
  img.onload = () => {
    if (state.live.playing && cam === state.live.cam && tok === liveTok
        && state.live.imgLive) {
      liveTimer = setTimeout(() => scheduleFrame(cam, tok), LIVE_POLL_MS);
    }
  };
  img.onerror = () => {
    if (state.live.playing && cam === state.live.cam && tok === liveTok
        && state.live.imgLive) {
      liveTimer = setTimeout(() => scheduleFrame(cam, tok), 3000);
    }
  };
  img.src = '/api/live/' + encodeURIComponent(cam) + '/latest.jpg?t=' + Date.now();
}

function stopStream() {
  state.live.playing = false;
  state.live.mode = '';
  state.live.imgLive = false;
  liveTok++;
  if (liveTimer) { clearTimeout(liveTimer); liveTimer = null; }
  if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  clearFrameWatch();
  if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
  if (offlineTimer) { clearInterval(offlineTimer); offlineTimer = null; }
  clearPaintCheck();
  clearStallWatch();
  if (curHls) { curHls.destroy(); curHls = null; }
  const video = $('#live-video');
  try { video.pause(); } catch (e) { /* ignore */ }
  try { video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; img.removeAttribute('src'); }
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.add('hidden');
  $('#live-sound').classList.add('hidden');
  hideSpinner();
  $('#live-status').textContent = '';
  resetZoom();                 // a fresh camera / restart starts at 1x, no stale pan
}

function resume() {
  if (!state.live.cam) return;
  startStream(state.live.cam);
}

/* -------- live control row extras: save image + zoom & pan ----------------
   The row above the frame holds (in DOM order): the camera name/picker (first),
   the sound toggle, save-image, the zoom group (- / + / reset), and the refresh
   (restart) button (last). Zoom scales whichever media is visible (the HLS
   <video> or the poster <img>) about the frame centre via a CSS transform; once
   zoomed the user can DRAG the picture to pan (desktop mouse wheel zooms too).
   Zoom is cosmetic - "save image" always captures the FULL current frame. */
const ZOOM_MIN = 1;
const ZOOM_MAX = 4;
const ZOOM_STEP = 1.5;        // per click / wheel notch
const zoom = { s: 1, tx: 0, ty: 0 };
let panPt = null;             // {id, x, y} while the user drags a zoomed frame

function liveStage() { return $('#live-stage'); }
/* True when there is actually a picture on screen to zoom (a decoding video or
   a loaded poster) - false for offline/paused states. */
function zoomable() {
  const v = $('#live-video'), im = $('#live-img');
  if (v && !v.classList.contains('hidden') && v.videoWidth > 0) return true;
  if (im && !im.classList.contains('hidden') && im.naturalWidth > 0) return true;
  return false;
}
/* Intrinsic size of the currently visible media (drives the pan limits). */
function mediaIntrinsic() {
  const v = $('#live-video'), im = $('#live-img');
  if (v && !v.classList.contains('hidden') && v.videoWidth > 0) {
    return { w: v.videoWidth, h: v.videoHeight };
  }
  if (im && im.naturalWidth > 0) return { w: im.naturalWidth, h: im.naturalHeight };
  return null;
}
function clampAxis(t, half) {
  return half > 0 ? Math.max(-half, Math.min(half, t)) : 0;
}
/* Keep the pan inside the scaled picture: a zoomed-in frame must always cover
   the whole stage, so the user can't drag past the content edge into blank. */
function constrainZoomPan() {
  const m = mediaIntrinsic(), st = liveStage();
  if (!m || !st) return;
  const W = st.clientWidth, H = st.clientHeight;
  if (!W || !H) return;
  const ar = m.w / m.h;
  const dw = Math.min(W, H * ar);   // object-fit: contain display box, pre-scale
  const dh = dw / ar;
  zoom.tx = clampAxis(zoom.tx, (dw * zoom.s - W) / 2);
  zoom.ty = clampAxis(zoom.ty, (dh * zoom.s - H) / 2);
}
function applyZoom() {
  const v = $('#live-video'), im = $('#live-img');
  const t = zoom.s > ZOOM_MIN
    ? 'translate(' + zoom.tx + 'px,' + zoom.ty + 'px) scale(' + zoom.s + ')'
    : 'none';
  if (v) v.style.transform = t;
  if (im) im.style.transform = t;
  const zi = $('#zoom-in'), zo = $('#zoom-out'), zr = $('#zoom-reset');
  if (zi) zi.disabled = zoom.s >= ZOOM_MAX;
  if (zo) zo.disabled = zoom.s <= ZOOM_MIN;
  if (zr) zr.disabled = zoom.s <= ZOOM_MIN;
  const st = liveStage();
  if (st) st.classList.toggle('zoomed', zoom.s > ZOOM_MIN);
}
function setZoom(s) {
  s = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, s));
  if (s <= ZOOM_MIN) { zoom.s = 1; zoom.tx = 0; zoom.ty = 0; }
  else { zoom.s = s; constrainZoomPan(); }
  applyZoom();
}
function zoomIn() { setZoom(zoom.s * ZOOM_STEP); }
function zoomOut() { setZoom(zoom.s / ZOOM_STEP); }
function resetZoom() { setZoom(ZOOM_MIN); }

/* Drag-to-pan: only while zoomed (and only when a frame is really on screen);
   pointer capture keeps the drag smooth even when it leaves the stage. */
function onPanStart(e) {
  if (zoom.s <= ZOOM_MIN || !zoomable()) return;
  if (e.target.closest && e.target.closest('.overlay, .spinner, button')) return;
  const st = liveStage(); if (!st) return;
  st.classList.add('panning');
  panPt = { id: e.pointerId, x: e.clientX, y: e.clientY };
  try { st.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  e.preventDefault();
}
function onPanMove(e) {
  if (!panPt || panPt.id !== e.pointerId) return;
  zoom.tx += e.clientX - panPt.x;
  zoom.ty += e.clientY - panPt.y;
  panPt.x = e.clientX; panPt.y = e.clientY;
  constrainZoomPan();
  applyZoom();
  e.preventDefault();
}
function onPanEnd(e) {
  if (!panPt || panPt.id !== e.pointerId) return;
  panPt = null;
  const st = liveStage(); if (st) st.classList.remove('panning');
}

/* Save the CURRENT frame as a JPEG download. Priority: the live <video> (drawn
   to a canvas at its intrinsic size - the best quality), then the poster <img>,
   then a fresh detect snapshot if nothing is on screen yet. */
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.rel = 'noopener';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
async function saveLiveImage() {
  const cam = state.live.cam;
  if (!cam) { toast('No camera selected'); return; }
  const btn = $('#live-shot');
  btn.disabled = true;
  const name = cam + '-' + new Date().toISOString().replace(/[:.]/g, '-') + '.jpg';
  const finish = (ok, msg) => {
    btn.disabled = false;
    toast(msg || (ok ? 'Image saved' : 'Could not save image'));
  };
  try {
    const v = $('#live-video'), im = $('#live-img');
    if (v && !v.classList.contains('hidden') && v.videoWidth > 0 && v.readyState >= 2) {
      const c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
      const blob = await new Promise(res => c.toBlob(res, 'image/jpeg', 0.92));
      if (!blob) throw new Error('frame capture returned no image');
      downloadBlob(blob, name); finish(true); return;
    }
    let src = (im && !im.classList.contains('hidden') && im.src) ? im.src : null;
    if (!src) src = '/api/live/' + encodeURIComponent(cam) + '/latest.jpg?t=' + Date.now();
    const resp = await fetch(src, { cache: 'no-store' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const blob = await resp.blob();
    if (!blob.size) throw new Error('empty image');
    downloadBlob(blob, name); finish(true);
  } catch (e) {
    finish(false, 'Save failed: ' + (e && e.message ? e.message : e));
  }
}

/* ---- wiring: control-row buttons + stage pan & wheel zoom ---- */
$('#live-shot').addEventListener('click', saveLiveImage);
$('#zoom-in').addEventListener('click', zoomIn);
$('#zoom-out').addEventListener('click', zoomOut);
$('#zoom-reset').addEventListener('click', resetZoom);
const stageEl = liveStage();
if (stageEl) {
  stageEl.addEventListener('pointerdown', onPanStart);
  stageEl.addEventListener('pointermove', onPanMove);
  stageEl.addEventListener('pointerup', onPanEnd);
  stageEl.addEventListener('pointercancel', onPanEnd);
  stageEl.addEventListener('wheel', (e) => {
    if (!zoomable()) return;
    e.preventDefault();
    setZoom(zoom.s * (e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP));
  }, { passive: false });
}
applyZoom();   // initial button disabled state (1x: zoom-out + reset disabled)

/* -------- camera switching: thumbnail row / modal, prev + next buttons ---- */
function camNames() {
  return state.cameras.map(c => c.name).filter(n => n);
}
function switchCam(step) {
  const names = camNames();
  if (names.length < 2) return;
  const i = names.indexOf(state.live.cam);
  const next = names[(i + step + names.length) % names.length];
  if (next && next !== state.live.cam) selectCam(next);
}
function rememberCam(cam) {
  state.live.cam = cam;                 // startStream also sets it (idempotent)
  try { localStorage.setItem('portal.lastCam', cam); } catch (e) { /* ignore */ }
}
/* Pick a camera from the thumbnail row / modal / prev-next buttons. Clicking
   the camera that is already streaming just closes the picker (no restart). */
function selectCam(cam) {
  if (!cam || camNames().indexOf(cam) < 0) return;
  closeCamModal();
  if (cam === state.live.cam && state.live.playing) return;
  rememberCam(cam);
  startStream(cam);
  syncCamUI();
}

/* -------- camera thumbnail cells (inline row + modal grid) -------- */
let thumbTs = 0;                       // bumped to refresh the latest.jpg thumbs
function camThumbSrc(cam) {
  return '/api/live/' + encodeURIComponent(cam) + '/latest.jpg?t=' + thumbTs;
}
function camCell(c) {
  const on = !!c.online;
  const name = esc(c.name);
  const cls = 'cam-item' + (on ? '' : ' offline');
  return '<button type="button" class="' + cls + '" data-cam="' + esc(c.name) +
         '" title="' + name + (on ? '' : ' (offline)') + '">' +
    '<span class="cam-thumb">' +
      '<img loading="lazy" src="' + camThumbSrc(c.name) + '" alt="' + name + '">' +
      '<span class="cam-off">no signal</span>' +
    '</span>' +
    '<span class="cam-name">' + name + '</span>' +
  '</button>';
}
function camCells() {
  return (state.cameras || []).map(camCell).join('');
}
function bindCamThumbErrors(root) {
  $$('.cam-thumb img', root).forEach(im =>
    im.addEventListener('error', () => im.classList.add('broken'), { once: true }));
}
function renderCamPicker() {
  thumbTs = Date.now();                 // fresh detect-frame thumbnails
  $('#cam-strip').innerHTML = camCells();
  $('#cam-grid').innerHTML = camCells();
  bindCamThumbErrors($('#cam-strip'));
  bindCamThumbErrors($('#cam-grid'));
  syncCamUI();
}
function syncCamUI() {
  const cam = state.live.cam || '';
  const p = $('#cam-picker');
  if (p) p.textContent = cam ? cam + '  \u25BE' : 'Select camera';
  const tag = $('#live-cam-tag');
  if (tag) tag.textContent = cam;
  $$('.cam-item').forEach(b => b.classList.toggle('active', b.dataset.cam === cam));
}
function openCamModal() {
  // Refresh the thumbnails + online tags whenever the picker is opened.
  thumbTs = Date.now();
  $('#cam-grid').innerHTML = camCells();
  bindCamThumbErrors($('#cam-grid'));
  syncCamUI();
  $('#cam-modal').classList.remove('hidden');
}
function closeCamModal() {
  $('#cam-modal').classList.add('hidden');
}

$('#cam-picker').addEventListener('click', openCamModal);
$('#cam-modal-close').addEventListener('click', closeCamModal);
$('#cam-modal').addEventListener('click', (e) => {
  if (e.target.closest('[data-cam-close]')) closeCamModal();   // backdrop tap
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeCamModal();
});
// Clicking a camera thumbnail in the inline row or the modal picks it.
$('#cam-strip').addEventListener('click', (e) => {
  const b = e.target.closest('.cam-item');
  if (b && b.dataset.cam) selectCam(b.dataset.cam);
});
$('#cam-grid').addEventListener('click', (e) => {
  const b = e.target.closest('.cam-item');
  if (b && b.dataset.cam) selectCam(b.dataset.cam);
});

$('#overlay-resume').addEventListener('click', resume);
// Restart the current live stream (retry HLS) WITHOUT reloading the page;
// also re-arms a stream after an idle pause or a snapshot fallback.
$('#live-refresh').addEventListener('click', () => {
  if (!state.live.cam) return;
  startStream(state.live.cam);
});
$('#live-sound').addEventListener('click', (e) => {
  e.preventDefault();
  toggleLiveSound();
});
// Prev/next camera (wrap around) - small buttons below the live frame.
$('#cam-prev').addEventListener('click', () => switchCam(-1));
$('#cam-next').addEventListener('click', () => switchCam(1));

/* ---------------- shared time-range + pagination helpers ---------------- */
const TIME_PRESETS = [
  ['', 'All time'], ['1h', 'Last 1 hour'], ['6h', 'Last 6 hours'],
  ['24h', 'Last 24 hours'], ['7d', 'Last 7 days'], ['30d', 'Last 30 days'],
  ['custom', 'Custom…'],
];
const TIME_WINDOW_S = { '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800, '30d': 2592000 };

function fillTimeSelect(sel) {
  sel.innerHTML = TIME_PRESETS.map(t =>
    '<option value="' + esc(t[0]) + '">' + esc(t[1]) + '</option>').join('');
}

// Wire one view's Time preset select + custom From/To row; any change auto-refreshes.
function wireTimeControls(prefix, onApply) {
  const timeSel = $('#' + prefix + '-time');
  const wrap = $('#' + prefix + '-custom-wrap');
  const fromEl = $('#' + prefix + '-from');
  const toEl = $('#' + prefix + '-to');
  fillTimeSelect(timeSel);
  const sync = () => {
    const custom = timeSel.value === 'custom';
    wrap.classList.toggle('hidden', !custom);
    if (!custom) { fromEl.value = ''; toEl.value = ''; }
  };
  timeSel.addEventListener('change', () => { sync(); onApply(); });
  fromEl.addEventListener('change', onApply);
  toEl.addEventListener('change', onApply);
  sync();
}

// Active time window -> {after, before} (epoch s). Filled custom From/To win;
// otherwise a preset is a trailing window ending now.
function timeRange(timeSel, fromEl, toEl) {
  const now = Math.floor(Date.now() / 1000);
  const r = {};
  const after = fromEl.value ? Math.floor(new Date(fromEl.value).getTime() / 1000) : null;
  const before = toEl.value ? Math.floor(new Date(toEl.value).getTime() / 1000) + 60 : null;
  if (after != null && !isNaN(after)) r.after = after;
  if (before != null && !isNaN(before)) r.before = before;
  if (r.after || r.before) return r;        // custom From/To present
  const key = timeSel.value;
  if (key && TIME_WINDOW_S[key]) return { after: now - TIME_WINDOW_S[key] };
  return {};
}

function hidePager(prefix) {
  const p = $('#' + prefix + '-pager');
  if (p) p.classList.add('hidden');
}
function renderPager(prefix, page, pages) {
  const pager = $('#' + prefix + '-pager');
  if (!pager) return;
  if (pages <= 1) { pager.classList.add('hidden'); return; }
  pager.classList.remove('hidden');
  const prev = $('#' + prefix + '-prev');
  const next = $('#' + prefix + '-next');
  if (prev) prev.disabled = page <= 1;
  if (next) next.disabled = page >= pages;
  const pn = $('#' + prefix + '-pageno');
  if (pn) pn.textContent = 'Page ' + page + ' / ' + pages;
}

/* ---------------- Frigate events & detections (paged + time-filtered) -------- */
const EV_PAGE = 24;
const EV_MAX = 5000;     // client-side paging cap for a single Frigate fetch
let evAll = [];          // current filter's full event array (newest first)
let evPage = 1;

$('#ev-refresh').addEventListener('click', loadEvents);
$('#ev-cam').addEventListener('change', loadEvents);
$('#ev-label').addEventListener('change', loadEvents);
wireTimeControls('ev', loadEvents);
$('#ev-prev').addEventListener('click', () => {
  if (evPage > 1) { evPage--; renderEvPage(); }
});
$('#ev-next').addEventListener('click', () => {
  if (evPage < Math.max(1, Math.ceil(evAll.length / EV_PAGE))) { evPage++; renderEvPage(); }
});

// Clicking a clip's play button swaps its thumbnail preview for the playing video.
$('#ev-list').addEventListener('click', (e) => {
  const btn = e.target.closest('.evmedia-play');
  if (!btn) return;
  const media = btn.parentElement;
  const img = media.querySelector('.evmedia-img');
  const vid = media.querySelector('.evmedia-video');
  if (!vid) return;
  if (img) img.classList.add('hidden');
  btn.classList.add('hidden');
  vid.classList.remove('hidden');
  vid.play().catch(() => {});
});

async function loadEvents() {
  const box = $('#ev-list');
  const st = $('#ev-status');
  st.textContent = 'Loading…';
  hidePager('ev');
  try {
    const { after, before } = timeRange($('#ev-time'), $('#ev-from'), $('#ev-to'));
    const p = new URLSearchParams({ limit: String(EV_MAX) });
    const cam = $('#ev-cam').value; if (cam) p.set('camera', cam);
    const lab = $('#ev-label').value.trim(); if (lab) p.set('label', lab);
    if (after) p.set('after', String(after));
    if (before) p.set('before', String(before));
    const events = await api('/api/events?' + p.toString());
    evAll = Array.isArray(events) ? events : [];
    evPage = 1;
    st.textContent = '';
    renderEvPage();
  } catch (e) {
    evAll = [];
    st.textContent = '';
    box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
    hidePager('ev');
  }
}

function renderEvPage() {
  const box = $('#ev-list');
  const pages = Math.max(1, Math.ceil(evAll.length / EV_PAGE));
  if (evPage > pages) evPage = pages;
  const slice = evAll.slice((evPage - 1) * EV_PAGE, evPage * EV_PAGE);
  $('#ev-status').textContent =
    (evAll.length ? evAll.length + ' event(s)' : '') +
    (evAll.length >= EV_MAX ? ' (older events omitted - narrow the time filter)' : '');
  renderEvents(slice, box);
  renderPager('ev', evPage, pages);
}

function renderEvents(events, box) {
  if (!Array.isArray(events) || !events.length) {
    box.innerHTML = '<div class="empty">No events</div>';
    return;
  }
  box.innerHTML = events.map(ev => {
    const lab = esc(ev.label || 'detection');
    const cam = esc(ev.camera || '');
    const score = ev.top_score != null ? ev.top_score : ev.score || 0;
    const id = encodeURIComponent(ev.id);
    const snapUrl = '/api/events/' + id + '/snapshot.jpg';
    const clipUrl = '/api/events/' + id + '/clip.mp4';
    const hasSnap = !!ev.has_snapshot;
    const hasClip = !!ev.has_clip;

    let media;
    if (hasClip) {
      // Thumbnail is the clip preview; the ▶ button swaps it for the playing video.
      media =
        '<div class="thumb evmedia">' +
          (hasSnap
            ? '<img class="evmedia-img" loading="lazy" src="' + snapUrl + '" alt="">'
            : '<div class="evmedia-nosnap" title="no preview"></div>') +
          '<button class="evmedia-play" type="button" aria-label="Play clip">&#9654;</button>' +
          '<video class="evmedia-video hidden" controls playsinline preload="none" ' +
            'src="' + clipUrl + '"></video>' +
        '</div>';
    } else {
      media =
        '<div class="thumb">' +
          (hasSnap
            ? '<img loading="lazy" src="' + snapUrl + '" alt="">'
            : '<div class="empty" style="padding:10px">no snapshot</div>') +
        '</div>';
    }

    return '<div class="card">' +
      media +
      '<div class="meta">' +
        '<span class="tag">' + lab + '</span>' +
        '<span class="tag">' + cam + '</span>' +
        '<span>score ' + Number(score).toFixed(2) + '</span>' +
        '<span class="muted">' + fmtDT(ev.start_time) + '</span>' +
      '</div>' +
    '</div>';
  }).join('');
}

/* ---------------- Firewatch fire detections & alerts (paged + time-filtered) -- */
const FW_PAGE = 24;
let fwPage = 1;
let fwPages = 1;

function reloadFire() {
  fwPage = 1;
  fwDets = {};
  loadFire();
}
$('#fw-refresh').addEventListener('click', reloadFire);
$('#fw-cam').addEventListener('change', reloadFire);
$('#fw-label').addEventListener('change', reloadFire);
$('#fw-alerted').addEventListener('change', reloadFire);
wireTimeControls('fw', reloadFire);
$('#fw-prev').addEventListener('click', () => {
  if (fwPage > 1) { fwPage--; loadFire(); }
});
$('#fw-next').addEventListener('click', () => {
  if (fwPage < fwPages) { fwPage++; loadFire(); }
});

async function loadFire() {
  const box = $('#fw-list');
  const st = $('#fw-status');
  st.textContent = 'Loading…';
  hidePager('fw');
  try {
    const { after, before } = timeRange($('#fw-time'), $('#fw-from'), $('#fw-to'));
    const p = new URLSearchParams({
      limit: String(FW_PAGE),
      offset: String((fwPage - 1) * FW_PAGE),
    });
    const cam = $('#fw-cam').value; if (cam) p.set('camera', cam);
    const lab = $('#fw-label').value; if (lab) p.set('label', lab);
    if ($('#fw-alerted').checked) p.set('alerted', '1');
    if (after) p.set('after', String(after));
    if (before) p.set('before', String(before));
    const data = await api('/api/fire/events?' + p.toString());
    const total = data.total || 0;
    fwPages = Math.max(1, Math.ceil(total / FW_PAGE));
    if (fwPage > fwPages) fwPage = fwPages;
    st.textContent = total ? total + ' record(s)' : '';
    box.innerHTML = '';
    if (!data.items.length) {
      box.innerHTML = '<div class="empty">No fire evidence</div>';
      hidePager('fw');
      return;
    }
    const frag = document.createElement('div');
    frag.innerHTML = data.items.map(fireCard).join('');
    $$('.fw-img', frag).forEach(img => {
      img.addEventListener('load', () => drawFireBoxes(img));
      if (img.complete) drawFireBoxes(img);
    });
    $$('.card', frag).forEach(c => box.appendChild(c));
    renderPager('fw', fwPage, fwPages);
  } catch (e) {
    st.textContent = '';
    box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
    hidePager('fw');
  }
}

function fireCard(f) {
  fwDets[f.id] = f.detections || [];
  const labels = [...new Set((f.detections || []).map(d => d.label))];
  const labelTags = labels.map(l => {
    const c = String(l).toLowerCase();
    return '<span class="tag ' + (c === 'smoke' ? 'smoke' : 'fire') + '">' + esc(l) + '</span>';
  }).join(' ');
  return '<div class="card">' +
    '<div class="thumb">' +
      (f.alerted ? '<span class="badge">alerted</span>' : '<span class="badge ok">detection</span>') +
      '<div class="fw-imgwrap">' +
        '<img class="fw-img" data-id="' + f.id + '" src="/api/fire/' + f.id + '/image.jpg" alt="fire evidence">' +
        '<div class="fw-overlays"></div>' +
      '</div>' +
    '</div>' +
    '<div class="meta">' +
      labelTags +
      '<span>best ' + Number(f.best_score).toFixed(2) + '</span>' +
      '<span class="tag">' + esc(f.camera) + '</span>' +
      '<span class="muted">' + esc(f.ts_utc || '') + '</span>' +
    '</div>' +
  '</div>';
}

function drawFireBoxes(img) {
  const id = img.dataset.id;
  const wrap = img.parentElement;
  const ov = wrap.querySelector('.fw-overlays');
  if (!wrap || !ov) return;
  const nw = img.naturalWidth || 640;
  const nh = img.naturalHeight || 360;
  ov.innerHTML = (fwDets[id] || []).map(d => {
    const x1 = d.x1 / nw * 100, y1 = d.y1 / nh * 100;
    const x2 = d.x2 / nw * 100, y2 = d.y2 / nh * 100;
    const cls = String(d.label).toLowerCase() === 'smoke' ? 'smoke' : '';
    return '<div class="fw-box ' + cls + '" style="left:' + x1 + '%;top:' + y1 + '%;' +
      'width:' + Math.max(0, x2 - x1) + '%;height:' + Math.max(0, y2 - y1) + '%"></div>';
  }).join('');
}

/* ---------------- start ---------------- */
boot();
