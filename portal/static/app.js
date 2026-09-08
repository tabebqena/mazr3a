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
  live: { cam: null, playing: false, mode: '' },  // mode: '' | 'hls' | 'snap'
  idleSec: 300,
};

const LIVE_POLL_MS = 1100;   // snapshot fallback rate (~1 fps, detect fps)
let liveTimer = null;
let idleTimer = null;
let liveTok = 0;             // guards stale async callbacks after switch/stop
let curHls = null;           // active hls.js instance (destroy on stop/switch)

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
  state.idleSec = state.settings.stream_idle_timeout_s || 300;
  await loadCameras();
  window.addEventListener('hashchange', onRoute);
  onRoute();
}

function onRoute() {
  const raw = (location.hash || '#/live').replace(/^#\//, '');
  const view = ['live', 'events', 'fire'].indexOf(raw) >= 0 ? raw : 'live';
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
  const names = state.cameras.map(c => c.name);

  const camOpts = names.map(n => {
    const c = state.cameras.find(x => x.name === n);
    const tag = c && !c.online ? ' (offline)' : '';
    return '<option value="' + esc(n) + '">' + esc(n) + tag + '</option>';
  }).join('');

  $('#cam-select').innerHTML = camOpts;
  $('#ev-cam').innerHTML = '<option value="">all cameras</option>' + camOpts;
  $('#fw-cam').innerHTML = '<option value="">all cameras</option>' + camOpts;

  const def = state.me.default_camera || data.default_camera
    || (state.cameras.find(c => c.enabled) || {}).name || names[0];
  state.live.cam = names.indexOf(def) >= 0 ? def : names[0] || null;
  if (state.live.cam) $('#cam-select').value = state.live.cam;
}

/* ---------------- Live view: HLS (go2rtc) via hls.js w/ snapshot fallback ----- */
// go2rtc in this Frigate 0.17.2 build has no working MSE-over-WS and no HTTP
// MSE endpoint; its tunnel-friendly live transport is HLS (master + ~0.5 s .ts
// segments), proxied SAME-ORIGIN through the portal at
//   /api/live/<cam>/hls/stream.m3u8?src=<cam>
// hls.js plays it (MSE inside the page); Safari plays it natively. Any failure
// falls back to the detect-snapshot proxy (Frigate /api/<cam>/latest.jpg).

function hlsFallback(cam, tok) {
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || state.live.mode !== 'hls') return;
  $('#live-status').textContent = 'HLS unavailable - snapshot mode (1 fps)';
  startSnapshot(cam, tok);
}

function resetIdle() {
  if (idleTimer) clearTimeout(idleTimer);
  if (!state.live.playing) return;
  idleTimer = setTimeout(onIdleTimeout, Math.max(30, state.idleSec) * 1000);
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
    if (state.live.playing) resetIdle();
  }, { passive: true }));

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
function showImage() {
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.remove('hidden');
}

function ensureLive() {
  if (!state.live.cam || state.live.playing) return;
  startStream(state.live.cam);
}

function startStream(cam) {
  stopStream();
  const tok = ++liveTok;
  state.live.cam = cam;
  state.live.playing = true;
  hideOverlay();
  showSpinner();            // spinner until the first live frame arrives
  startHls(cam, tok);
  resetIdle();
}

function startHls(cam, tok) {
  state.live.mode = 'hls';
  showVideo();
  const video = $('#live-video');
  // Same-origin HLS through the portal (behind the session cookie):
  // https://<portal>/api/live/<cam>/hls/stream.m3u8?src=<cam>
  const url = '/api/live/' + encodeURIComponent(cam) +
    '/hls/stream.m3u8?src=' + encodeURIComponent(cam);
  $('#live-status').textContent = 'Connecting ' + cam + '…';

  if (window.Hls && Hls.isSupported()) {
    // hls.js -> MediaSource in the page (Chrome/Firefox/Edge etc.)
    const hls = curHls = new Hls({
      enableWorker: true,
      lowLatencyMode: true,
      backBufferLength: 30,
      maxLiveSyncPlaybackRate: 1.5,
    });
    hls.loadSource(url);
    hls.attachMedia(video);
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      if (state.live.playing && tok === liveTok && cam === state.live.cam) {
        $('#live-status').textContent = 'Live (HLS)';
        hideSpinner();
        video.play().catch(() => {});
      }
    });
    hls.on(Hls.Events.ERROR, (e, data) => {
      if (!data || !data.fatal) return;
      if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad();
      else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError();
      else hlsFallback(cam, tok);
    });
    return;
  }

  if (video.canPlayType('application/vnd.apple.mpegurl')) {
    // Safari / iOS native HLS
    const onErr = () => hlsFallback(cam, tok);
    video.addEventListener('error', onErr, { once: true });
    video.src = url;
    video.play().then(() => {
      if (state.live.playing && tok === liveTok && cam === state.live.cam) {
        $('#live-status').textContent = 'Live (HLS)';
        hideSpinner();
      }
    }).catch(() => hlsFallback(cam, tok));
    return;
  }

  $('#live-status').textContent = 'HLS unsupported in this browser';
  startSnapshot(cam, tok);
}

function startSnapshot(cam, tok) {
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam) return;
  state.live.mode = 'snap';
  showSpinner();            // spinner until the first detect frame loads
  showImage();
  $('#live-status').textContent = 'Live (snapshot ~1 fps)';
  scheduleFrame(cam, tok);
}

function scheduleFrame(cam, tok) {
  if (!state.live.playing || cam !== state.live.cam || tok !== liveTok
      || state.live.mode !== 'snap') return;
  const img = $('#live-img');
  img.onload = () => {
    if (state.live.playing && cam === state.live.cam && tok === liveTok
        && state.live.mode === 'snap') {
      hideSpinner();
      liveTimer = setTimeout(() => scheduleFrame(cam, tok), LIVE_POLL_MS);
    }
  };
  img.onerror = () => {
    if (state.live.playing && cam === state.live.cam && tok === liveTok
        && state.live.mode === 'snap') {
      liveTimer = setTimeout(() => scheduleFrame(cam, tok), 3000);
    }
  };
  img.src = '/api/live/' + encodeURIComponent(cam) + '/latest.jpg?t=' + Date.now();
}

function stopStream() {
  state.live.playing = false;
  state.live.mode = '';
  liveTok++;
  if (liveTimer) { clearTimeout(liveTimer); liveTimer = null; }
  if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  if (curHls) { curHls.destroy(); curHls = null; }
  const video = $('#live-video');
  try { video.pause(); } catch (e) { /* ignore */ }
  try { video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; img.removeAttribute('src'); }
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.add('hidden');
  hideSpinner();
  $('#live-status').textContent = '';
}

function resume() {
  if (!state.live.cam) return;
  startStream(state.live.cam);
}

$('#cam-select').addEventListener('change', (e) => startStream(e.target.value));
$('#overlay-resume').addEventListener('click', resume);

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
