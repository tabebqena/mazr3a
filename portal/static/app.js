/* mazr3a CCTV portal SPA - vanilla JS.
   Views: Live (single camera + idle time watch), Frigate Events & detections,
   Firewatch fire detections & alerts. Talks to the portal /api/* (same origin,
   session cookie). Live video is a direct go2rtc MSE URL (Access-gated at the
   Cloudflare Tunnel). */
'use strict';

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

const state = {
  me: null,
  settings: null,
  cameras: [],
  live: { cam: null, playing: false },
  idleSec: 300,
};

let fwDets = {};   // fire frame id -> detections (for box overlay)

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
  const stored = localStorage.getItem('portal.idleSec');
  state.idleSec = Number(stored) || state.settings.stream_idle_timeout_s || 300;
  $('#idle-input').value = state.idleSec;
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
  else if (view === 'fire') loadFire();
}

async function loadCameras() {
  const data = await api('/api/cameras');
  state.cameras = data.cameras || [];
  const names = state.cameras.map(c => c.name);

  const camOpts = names.map(n => {
    const c = state.cameras.find(x => x.name === n);
    const tag = c && !c.online ? ' (offline)' : '';
    return `<option value="${esc(n)}">${esc(n)}${tag}</option>`;
  }).join('');

  $('#cam-select').innerHTML = camOpts;
  $('#ev-cam').innerHTML = '<option value="">all cameras</option>' + camOpts;
  $('#fw-cam').innerHTML = '<option value="">all cameras</option>' + camOpts;

  const def = state.me.default_camera || data.default_camera
    || (state.cameras.find(c => c.enabled) || {}).name || names[0];
  state.live.cam = names.indexOf(def) >= 0 ? def : names[0] || null;
  if (state.live.cam) $('#cam-select').value = state.live.cam;
}

/* ---------------- Live view + MSE + idle time watch ---------------- */
const player = new G2MSEPlayer($('#live-video'));
player.onerror = (e) => {
  showOverlay('Live stream unavailable (' + e.message + ')');
};
player.onstatus = (m) => {
  const st = $('#live-status');
  if (m) st.textContent = m;
};

let idleTimer = null;
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
  showOverlay('Stream paused after ' + txt + ' without activity.');
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

function ensureLive() {
  if (!state.live.cam) return;
  if (state.live.playing && player.connected) return; // already streaming
  startStream(state.live.cam);
}
function startStream(cam) {
  stopStream();
  state.live.cam = cam;
  state.live.playing = true;
  hideOverlay();
  $('#live-status').textContent = 'Connecting ' + cam + '…';
  api('/api/stream-url/' + encodeURIComponent(cam)).then(r => {
    player.play(r.url);
    resetIdle();
  }).catch(e => {
    $('#live-status').textContent = '';
    showOverlay('Cannot start stream: ' + e.message);
  });
}
function stopStream() {
  state.live.playing = false;
  if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  player.stop();
  $('#live-status').textContent = '';
}
function resume() {
  if (!state.live.cam) return;
  startStream(state.live.cam);
}

$('#cam-select').addEventListener('change', (e) => startStream(e.target.value));
$('#idle-input').addEventListener('change', (e) => {
  let v = Math.max(30, parseInt(e.target.value, 10) || state.settings.stream_idle_timeout_s || 300);
  state.idleSec = v;
  e.target.value = v;
  try { localStorage.setItem('portal.idleSec', String(v)); } catch (err) { /* ignore */ }
  if (state.live.playing) resetIdle();
});
$('#overlay-resume').addEventListener('click', resume);

/* ---------------- Frigate events & detections ---------------- */
$('#ev-refresh').addEventListener('click', loadEvents);
$('#ev-cam').addEventListener('change', loadEvents);
$('#ev-label').addEventListener('change', loadEvents);

async function loadEvents() {
  const box = $('#ev-list');
  const st = $('#ev-status');
  st.textContent = 'Loading…';
  try {
    const p = new URLSearchParams({ limit: '60' });
    const cam = $('#ev-cam').value; if (cam) p.set('camera', cam);
    const lab = $('#ev-label').value.trim(); if (lab) p.set('label', lab);
    const events = await api('/api/events?' + p.toString());
    st.textContent = '';
    renderEvents(events, box);
  } catch (e) {
    st.textContent = '';
    box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
  }
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
    const thumb = ev.has_snapshot
      ? `<img loading="lazy" src="/api/events/${ev.id}/snapshot.jpg" alt="">`
      : '<div class="empty" style="padding:10px">no snapshot</div>';
    const clip = ev.has_clip
      ? `<video class="media" controls preload="none" src="/api/events/${ev.id}/clip.mp4"></video>`
      : '';
    return `<div class="card">
      <div class="thumb">${thumb}</div>
      <div class="meta">
        <span class="tag">${lab}</span>
        <span class="tag">${cam}</span>
        <span>score ${Number(score).toFixed(2)}</span>
        <span class="muted">${fmtDT(ev.start_time)}</span>
      </div>${clip}
    </div>`;
  }).join('');
}

/* ---------------- Firewatch fire detections & alerts ---------------- */
$('#fw-refresh').addEventListener('click', () => loadFire());
$('#fw-cam').addEventListener('change', () => loadFire());
$('#fw-label').addEventListener('change', () => loadFire());
$('#fw-alerted').addEventListener('change', () => loadFire());

let fwOffset = 0;
async function loadFire(reset) {
  if (reset !== false) { reset = true; fwOffset = 0; fwDets = {}; }
  const box = $('#fw-list');
  const st = $('#fw-status');
  if (reset) st.textContent = 'Loading…';
  try {
    const p = new URLSearchParams({ limit: '24', offset: String(fwOffset) });
    const cam = $('#fw-cam').value; if (cam) p.set('camera', cam);
    const lab = $('#fw-label').value; if (lab) p.set('label', lab);
    if ($('#fw-alerted').checked) p.set('alerted', '1');
    const data = await api('/api/fire/events?' + p.toString());
    st.textContent = data.total ? `${data.total} record(s)` : '';
    if (reset) box.innerHTML = '';
    if (!data.items.length) {
      if (reset) box.innerHTML = '<div class="empty">No fire evidence</div>';
      removeLoadMore();
      return;
    }
    const frag = document.createElement('div');
    frag.innerHTML = data.items.map(fireCard).join('');
    $$('.fw-img', frag).forEach(img => {
      img.addEventListener('load', () => drawFireBoxes(img));
      if (img.complete) drawFireBoxes(img);
    });
    $$('.card', frag).forEach(c => box.appendChild(c));
    fwOffset += data.items.length;
    ensureLoadMore(fwOffset < data.total);
  } catch (e) {
    st.textContent = '';
    if (reset) box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
  }
}

function fireCard(f) {
  fwDets[f.id] = f.detections || [];
  const labels = [...new Set((f.detections || []).map(d => d.label))];
  const labelTags = labels.map(l => {
    const c = String(l).toLowerCase();
    return `<span class="tag ${c === 'smoke' ? 'smoke' : 'fire'}">${esc(l)}</span>`;
  }).join(' ');
  return `<div class="card">
    <div class="thumb">
      ${f.alerted ? '<span class="badge">alerted</span>' : '<span class="badge ok">detection</span>'}
      <div class="fw-imgwrap">
        <img class="fw-img" data-id="${f.id}" src="/api/fire/${f.id}/image.jpg" alt="fire evidence">
        <div class="fw-overlays"></div>
      </div>
    </div>
    <div class="meta">
      ${labelTags}
      <span>best ${Number(f.best_score).toFixed(2)}</span>
      <span class="tag">${esc(f.camera)}</span>
      <span class="muted">${esc(f.ts_utc || '')}</span>
    </div>
  </div>`;
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
    return `<div class="fw-box ${cls}" style="left:${x1}%;top:${y1}%;` +
      `width:${Math.max(0, x2 - x1)}%;height:${Math.max(0, y2 - y1)}%"></div>`;
  }).join('');
}

function ensureLoadMore(show) {
  let lm = $('#fw-loadmore');
  if (show) {
    if (!lm) {
      lm = document.createElement('button');
      lm.id = 'fw-loadmore';
      lm.className = 'loadmore';
      lm.textContent = 'Load more';
      lm.addEventListener('click', () => loadFire(false));
      $('#fw-list').after(lm);
    }
  } else {
    removeLoadMore();
  }
}
function removeLoadMore() {
  const lm = $('#fw-loadmore');
  if (lm) lm.remove();
}

/* ---------------- start ---------------- */
boot();
