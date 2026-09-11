 /* mazr3a CCTV portal SPA - vanilla JS.
   Views: Login, Live (single camera; go2rtc MSE over WebSocket - the same
   transport Frigate's own UI uses - proxied same-origin through the portal,
   with an hls.js HLS fallback and a detect-snapshot poster, idle time watch),
   Frigate Events & detections, Firewatch fire detections & alerts. Talks to
   the portal /api/* (same origin, session cookie). */
'use strict';

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

const state = {
  me: null,
  settings: null,
  cameras: [],
  usage: null,       // caller's own bandwidth breakdown (from /api/usage)
  users: [],         // admin (Manage tab): every account (from /api/users)
  manageTarget: null,// Manage tab: username selected for editing
  userTarget: null,  // Account tab: always the signed-in user
  userPhoto: '',     // Account tab: pending profile photo data URL
  notifications: [], // notification feed, newest first (/api/notifications)
  notifyUnread: 0,   // unread count shown on the nav badge
  lastNotifiedId: null, // highest id already ALERTED (the poller's baseline)
  booted: false,     // guard: continueBoot() runs once per signed-in session
  live: { cam: null, playing: false, mode: '', sound: false, imgLive: false,
          revealed: false },   // true once this live attempt dropped the poster
  // imgLive: the <img> is the poster shown while a live attempt is starting
  //          (or the frozen last frame while an online camera auto-retries).
  // mode: '' | 'mse' | 'hls' | 'offline'
  idleSec: 30,
};

const LIVE_POLL_MS = 1100;   // snapshot fallback rate (~1 fps, detect fps)
const SNAPSHOT_RETRY_MS = 15000;  // auto-retry interval: snapshot -> live
const OFFLINE_CHECK_MS = 8000;    // how often to re-check an offline camera
const LIVE_SOUND_KEY = 'portal.liveSound';  // localStorage: remembered voice on/off
// Live transport: go2rtc MSE over WebSocket is PRIMARY (robust - the same
// transport Frigate's own UI uses). hls.js HLS is a FALLBACK for browsers with
// no MediaSource. The watchdogs below operate on the <video> element + decoded
// frame counters, so they apply to BOTH transports.
const MSE_SUPPORTED = !!(window.MediaSource && window.MediaSource.isTypeSupported);
const MSE_BACK_BUFFER_S = 15;     // trim media behind the playhead (bound memory)
const HLS_PAINT_WAIT_MS = 3500;   // post-reveal: claim Live only once a real
                                  // video frame has been decoded (frame-based)
// Retry when media buffered but no video frame ever rendered (blackRevert path).
const HLS_VIDEO_RETRY_MS = 6000;
// Post-"Live" watchdog: if no NEW video frame decodes for this long while Live
// is claimed, tear down and restart the transport (fresh go2rtc session)
// instead of sitting on a black/frozen picture.
const HLS_STALL_WATCH_MS = 4000;
// HLS fallback support (hls.js MSE, or native Safari HLS where MediaSource is
// absent). MSE is preferred; HLS is only used when MSE is unavailable.
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
let curHls = null;           // active hls.js instance (HLS fallback; destroy on stop/switch)
let curMse = null;           // active go2rtc MSE session {ms,url,ws,buf,mime,queue,...}
let frameWatchTimer = null;  // watchdog: give up waiting for the first media
let retryTimer = null;       // auto-retry: snapshot fallback -> HLS live
let offlineTimer = null;     // periodic re-check while a camera is offline
let paintCheck = null;       // post-reveal: confirm frames really render
let stallWatch = null;       // post-Live: restart if no new frame (black/frozen)

let fwDets = {};             // fire frame id -> detections (for box overlay)
let fwMeta = {};             // fire frame id -> full record (for the lightbox)

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
/* True while a Live transport attempt is active (MSE primary or HLS fallback).
   The shared teardown/watchdog guards use this so they work for either. */
function isLiveTrying() {
  const m = state.live.mode;
  return m === 'mse' || m === 'hls';
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
  document.body.classList.remove('authed');
  $('#app').classList.add('hidden');
  $('#login-view').classList.remove('hidden');
}
function hideLogin() {
  document.body.classList.add('authed');
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

/* Sign out. The topbar #logout-btn and the collapsed-menu #nav-logout entry
   share this; both close the menu first. */
async function doLogout() {
  setNavOpen(false);
  stopStream();
  try { await api('/api/logout', { method: 'POST' }); } catch (e) { /* ignore */ }
  showLogin();
}
$('#logout-btn').addEventListener('click', doLogout);
$('#nav-logout').addEventListener('click', doLogout);

/* ---------------- collapsed nav (phones) ----------------
   On phones the tab bar is collapsed into a dropdown toggled by #nav-toggle
   (CSS shows that button only in the phone layout; on tablet/desktop the nav
   is an inline row and the button is hidden). This applies to EVERY tab/view
   and both orientations. The menu closes on tab pick, route change, outside
   click/tap, Escape, and an orientation change. */
const navToggle = $('#nav-toggle');
function setNavOpen(open) {
  if (!navToggle) return;
  document.body.classList.toggle('nav-open', !!open);
  navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
}
if (navToggle) {
  navToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    setNavOpen(!document.body.classList.contains('nav-open'));
  });
  // Picking a tab closes the menu (onRoute also closes it on the hash change).
  $('#nav').addEventListener('click', (e) => {
    if (e.target.closest('a')) setNavOpen(false);
  });
  document.addEventListener('click', (e) => {
    if (!document.body.classList.contains('nav-open')) return;
    if (!e.target.closest('#nav') && !e.target.closest('#nav-toggle')) {
      setNavOpen(false);
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setNavOpen(false);
  });
  window.addEventListener('orientationchange', () => setNavOpen(false));
}

/* ---------------- PWA install prompt (Android home-screen app) ----------------
   Chrome fires `beforeinstallprompt` when the portal is installable (manifest +
   icons over HTTPS) and is NOT already installed. We stash that event, reveal
   the in-app Install UI, and call prompt() only from a user gesture. Nothing
   here is needed for the portal to work - the browser URL stays fully usable. */
const INSTALL_DISMISS_KEY = 'portal.installDismissed';  // localStorage: user said no
let deferredInstallPrompt = null;   // the stashed beforeinstallprompt event

function isStandalone() {
  // Android WebAPK/PWA runs without browser chrome -> nothing to install.
  try {
    return window.matchMedia('(display-mode: standalone)').matches
      || window.matchMedia('(display-mode: fullscreen)').matches
      || window.navigator.standalone === true
      || (document.referrer || '').startsWith('android-app://');
  } catch (e) { return false; }
}

function installDismissed() {
  try { return localStorage.getItem(INSTALL_DISMISS_KEY) === '1'; }
  catch (e) { return false; }
}

// Show/hide BOTH install affordances (banner + collapsed-menu entry) together.
function setInstallUi(visible) {
  const banner = $('#install-banner');
  if (banner) banner.classList.toggle('hidden', !visible);
  const navBtn = $('#nav-install');
  if (navBtn) navBtn.classList.toggle('hidden', !visible);
}

async function promptInstall() {
  if (!deferredInstallPrompt) return;
  const ev = deferredInstallPrompt;
  deferredInstallPrompt = null;   // a prompt event can only be used ONCE
  setInstallUi(false);
  try {
    ev.prompt();
    await ev.userChoice;          // { outcome: 'accepted' | 'dismissed' }
  } catch (e) { /* ignore - the browser menu still offers installation */ }
}

function initInstallPrompt() {
  setInstallUi(false);
  if (isStandalone()) return;     // already an app: never offer installation
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();           // suppress Chrome's own mini-infobar
    deferredInstallPrompt = e;
    if (installDismissed()) return;
    setInstallUi(true);
  });
  const navBtn = $('#nav-install');
  if (navBtn) navBtn.addEventListener('click', () => {
    setNavOpen(false);
    promptInstall();
  });
  const go = $('#install-banner-go');
  if (go) go.addEventListener('click', promptInstall);
  const x = $('#install-banner-x');
  if (x) x.addEventListener('click', () => {
    try { localStorage.setItem(INSTALL_DISMISS_KEY, '1'); } catch (e) {}
    setInstallUi(false);
  });
  window.addEventListener('appinstalled', () => {
    deferredInstallPrompt = null;
    setInstallUi(false);
    try { localStorage.removeItem(INSTALL_DISMISS_KEY); } catch (e) {}
  });
}

/* ---------------- notifications (feed + browser alerts + access gate) --------
   The portal keeps a server-side notification FEED (portal/notifstore.py): a
   portal background watcher records ONE item per new Firewatch fire alert and an
   admin can publish a system message. This SPA:
     * GATES the whole app on the browser Notification permission - the
       requirement is that the user must ALLOW notifications to run the app;
     * polls the feed for NEW items and raises a real browser/OS notification for
       each, via the notification-only service worker /sw.js (REQUIRED on Android
       Chrome, where `new Notification()` throws `Illegal constructor`);
     * shows the feed in the Notifications tab with an unread nav badge.
   There is deliberately NO Web Push: the feed is derived by polling. */
const NOTIFY_POLL_MS = 30000;   // look for new notifications every 30 s
const NOTIFY_PAGE = 50;         // feed page size
// Icon for the OS notification: reuse the fingerprinted apple-touch icon the
// page already links, so the icon matches the installed app and no hashed URL
// is hard-coded here.
const NOTIFY_ICON = (function () {
  const link = document.querySelector('link[rel="apple-touch-icon"]');
  return (link && link.getAttribute('href')) || '/favicon.ico';
})();
let notifyTimer = null;

function notificationSupported() { return ('Notification' in window); }
function notificationPermission() {
  if (!notificationSupported()) return 'unsupported';
  try { return Notification.permission; } catch (e) { return 'unsupported'; }
}
function notificationGranted() { return notificationPermission() === 'granted'; }

/* ---- access gate: the portal requires notification permission ---- */
function isAndroidUA() { return /android/i.test(navigator.userAgent || ''); }
function isIOSUA() {
  return /iphone|ipad|ipod/i.test(navigator.userAgent || '')
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}
/* Platform-specific "how to allow it" text for the gate's DENIED state. */
function notifyGateInstructions(perm) {
  if (perm === 'unsupported') {
    return 'This browser cannot show notifications, so the portal cannot run ' +
      'here. Open it in a supported browser (Chrome, Edge, Firefox or Safari).';
  }
  if (perm === 'denied') {
    if (isIOSUA()) {
      return 'Notifications are blocked. Open <b>Settings \u2192 mazr3a</b> ' +
        '(or <b>Settings \u2192 Safari</b>) <b>\u2192 Notifications</b> and ' +
        'turn them <b>on</b>, then tap Retry.';
    }
    if (isAndroidUA()) {
      return 'Notifications are blocked. In <b>Chrome</b> open <b>\u22ee ' +
        '\u2192 Settings \u2192 Site settings \u2192 Notifications</b>, find ' +
        'this site and choose <b>Allow</b>. For the installed app: <b>Android ' +
        'Settings \u2192 Apps \u2192 mazr3a \u2192 Notifications \u2192 ' +
        'Allow</b>. Then tap Retry.';
    }
    return 'Notifications are blocked. Click the <b>lock / \u24d8 icon</b> in ' +
      'the address bar \u2192 <b>Site settings \u2192 Notifications \u2192 ' +
      'Allow</b>, then click Retry.';
  }
  return 'This app needs your permission to show notifications. Tap ' +
    '<b>Enable notifications</b> and choose <b>Allow</b>.';
}
function showNotifyGate() {
  const gate = $('#notify-gate');
  if (!gate) return;
  $('#app').classList.add('hidden');   // the gate REPLACES the dashboard
  const perm = notificationPermission();
  const blocked = perm === 'denied' || perm === 'unsupported';
  $('#nt-gate-title').textContent = blocked
    ? 'Notifications are blocked' : 'Notifications required';
  $('#nt-gate-msg').textContent = blocked
    ? 'The portal cannot run until notifications are allowed.'
    : 'Allow notifications to continue to the portal.';
  $('#nt-gate-steps').innerHTML = notifyGateInstructions(perm);
  const btn = $('#nt-gate-btn');
  btn.classList.toggle('hidden', blocked);
  btn.disabled = false;
  $('#nt-gate-retry').classList.toggle('hidden', !blocked);
  gate.classList.remove('hidden');
}
function hideNotifyGate() {
  const gate = $('#notify-gate');
  if (gate) gate.classList.add('hidden');
  const app = $('#app');
  if (app) app.classList.remove('hidden');
}
async function requestNotificationAccess() {
  if (!notificationSupported()) { showNotifyGate(); return; }
  let perm;
  try { perm = await Notification.requestPermission(); }
  catch (e) { perm = notificationPermission(); }
  if (perm === 'granted') { hideNotifyGate(); await continueBoot(); }
  else showNotifyGate();
}
/* Gate wiring runs at MODULE LOAD (not from boot): the buttons must work while
   the gate is up, which is BEFORE continueBoot() ever runs. */
function initNotifyGate() {
  const btn = $('#nt-gate-btn');
  if (btn) btn.addEventListener('click', () => {
    btn.disabled = true;
    requestNotificationAccess();
  });
  const retry = $('#nt-gate-retry');
  if (retry) retry.addEventListener('click', () => {
    // The user changed the permission in browser/OS settings: re-read it.
    if (notificationGranted()) { hideNotifyGate(); continueBoot(); }
    else showNotifyGate();
  });
}
initNotifyGate();

/* ---- service worker + browser notifications ---- */
async function registerNotifyWorker() {
  if (!('serviceWorker' in navigator)) return null;
  try { return await navigator.serviceWorker.register('/sw.js'); }
  catch (e) { return null; }
}
async function showBrowserNotification(n) {
  if (!notificationGranted()) return;
  const opts = {
    body: n.body || '',
    tag: 'mazr3a-' + n.id,
    icon: NOTIFY_ICON,
    badge: NOTIFY_ICON,
    data: { url: n.url || '#/notifications' },
  };
  const title = n.title || 'mazr3a CCTV';
  if ('serviceWorker' in navigator) {
    try {
      // `new Notification()` is desktop-only; never hang if the worker did not
      // register - race the readiness promise with a short timeout.
      const reg = await Promise.race([
        navigator.serviceWorker.ready,
        new Promise(res => setTimeout(() => res(null), 3000)),
      ]);
      if (reg && reg.showNotification) {
        await reg.showNotification(title, opts);
        return;
      }
    } catch (e) { /* fall through to the constructor */ }
  }
  try { new Notification(title, opts); } catch (e) { /* desktop fallback only */ }
}

/* ---- feed ---- */
function updateNotifyBadge() {
  const b = $('#nav-notify-badge');
  if (!b) return;
  const n = Number(state.notifyUnread) || 0;
  if (n > 0) {
    b.textContent = n > 99 ? '99+' : String(n);
    b.classList.remove('hidden');
  } else {
    b.classList.add('hidden');
  }
}
function notifCard(n) {
  return '<button type="button" class="nt-card' + (n.read ? '' : ' unread') +
    '" data-notif-id="' + Number(n.id) + '"' +
    (n.url ? ' data-notif-url="' + esc(n.url) + '"' : '') + '>' +
    '<span class="nt-dot" aria-hidden="true"></span>' +
    '<span class="nt-body">' +
      '<span class="nt-title">' + esc(n.title || '') + '</span>' +
      (n.body ? '<span class="nt-text">' + esc(n.body) + '</span>' : '') +
      '<span class="nt-meta">' + esc(n.kind || '') +
        (n.camera ? ' \u00b7 ' + esc(n.camera) : '') +
        ' \u00b7 ' + esc(fmtDT(n.created_at)) + '</span>' +
    '</span>' +
  '</button>';
}
function renderNotifications() {
  const box = $('#nt-list');
  if (!box) return;
  const items = state.notifications || [];
  if (!items.length) {
    box.innerHTML = '<div class="empty">No notifications yet.</div>';
    return;
  }
  box.innerHTML = items.map(notifCard).join('');
}
async function refreshNotifications() {
  const adm = $('#nt-admin');
  if (adm) adm.classList.toggle('hidden', !isAdmin());
  let data;
  try { data = await api('/api/notifications?limit=' + NOTIFY_PAGE); }
  catch (e) { return; }
  state.notifications = data.items || [];
  state.notifyUnread = Number(data.unread) || 0;
  // Baseline the poller on the FIRST load so the existing backlog is NOT
  // re-raised as a burst of OS notifications.
  const latest = Number(data.latest_id) || 0;
  state.lastNotifiedId = state.lastNotifiedId == null
    ? latest : Math.max(state.lastNotifiedId, latest);
  updateNotifyBadge();
  renderNotifications();
}
async function pollNotifications() {
  if (!notificationGranted()) return;
  let data;
  try {
    data = await api('/api/notifications?after_id=' +
      encodeURIComponent(state.lastNotifiedId == null ? 0 : state.lastNotifiedId));
  } catch (e) { return; }
  const items = data.items || [];
  if (items.length) {
    state.lastNotifiedId = Math.max(
      state.lastNotifiedId || 0, Number(data.latest_id) || 0);
    items.forEach(showBrowserNotification);   // one OS alert per new item
    state.notifications = items.slice().reverse()
      .concat(state.notifications || []).slice(0, NOTIFY_PAGE);
  }
  state.notifyUnread = Number(data.unread) || 0;
  updateNotifyBadge();
  if (items.length) renderNotifications();
}
function initNotifications() {
  registerNotifyWorker();
  refreshNotifications();
  if (notifyTimer) clearInterval(notifyTimer);
  notifyTimer = setInterval(() => {
    if (!document.hidden) pollNotifications();
  }, NOTIFY_POLL_MS);
}
function loadNotifications() { return refreshNotifications(); }

async function markNotifications(ids, all) {
  return api('/api/notifications/read', {
    method: 'POST',
    body: all ? { all: true } : { ids: ids },
  });
}
$('#nt-refresh').addEventListener('click', refreshNotifications);
$('#nt-readall').addEventListener('click', async () => {
  try {
    const r = await markNotifications([], true);
    state.notifyUnread = Number(r.unread) || 0;
  } catch (e) { /* leave the badge as-is */ }
  (state.notifications || []).forEach(n => { n.read = true; });
  updateNotifyBadge();
  renderNotifications();
});
$('#nt-list').addEventListener('click', async (e) => {
  const card = e.target.closest('.nt-card');
  if (!card) return;
  const id = Number(card.dataset.notifId);
  try {
    const r = await markNotifications([id], false);
    state.notifyUnread = Number(r.unread) || 0;
  } catch (e) { /* ignore */ }
  const item = (state.notifications || []).find(n => Number(n.id) === id);
  if (item) item.read = true;
  card.classList.remove('unread');
  updateNotifyBadge();
  const url = card.dataset.notifUrl || '';
  if (url && url !== location.hash) location.hash = url;
});
$('#nt-pub').addEventListener('click', async () => {
  const msg = $('#nt-msg');
  const title = ($('#nt-pub-title').value || '').trim();
  if (!title) { msg.textContent = 'Title is required.'; return; }
  msg.textContent = '';
  const btn = $('#nt-pub');
  btn.disabled = true;
  try {
    await api('/api/admin/notifications', { method: 'POST', body: {
      title: title,
      body: ($('#nt-pub-body').value || '').trim(),
      url: '#/notifications',
    }});
    $('#nt-pub-title').value = '';
    $('#nt-pub-body').value = '';
    msg.textContent = 'Published.';
    await pollNotifications();      // surface it immediately as an OS alert
    await refreshNotifications();
  } catch (e) {
    msg.textContent = 'Could not publish: ' + e.message;
  } finally {
    btn.disabled = false;
  }
});

/* ---------------- boot / router ---------------- */
async function boot() {
  try {
    state.me = await api('/api/me');
  } catch (e) { return; }
  hideLogin();
  // The signed-in username: inside the collapsed menu on phones (#nav-user) and
  // in the userbox of the horizontal layout (#user-chip). Both are CLICKABLE
  // links to the Account (User) tab. Guarded so a missing node can never break
  // boot.
  const navUser = $('#nav-user');
  if (navUser) navUser.textContent = state.me.username;
  const userChip = $('#user-chip');
  if (userChip) userChip.textContent = state.me.username;
  // Nav visibility: the admin-only Debug/Manage links, plus each `tab_*` a
  // non-admin was granted (empty = none). The server remains the real gate.
  applyNavPermissions();
  // The app REQUIRES notification access to run. Until the browser Notification
  // permission is GRANTED the dashboard stays hidden behind the gate, whose
  // buttons call continueBoot() once it is allowed.
  if (!notificationGranted()) { showNotifyGate(); return; }
  await continueBoot();
}

/* Everything AFTER the notification access gate. Guarded by state.booted so
   granting permission in the gate can never start the app twice. */
async function continueBoot() {
  if (state.booted) return;
  state.booted = true;
  state.settings = await api('/api/settings');
  // Idle stop is set by an admin in portal.conf (STREAM_IDLE_TIMEOUT_S); there is
  // no in-UI control, so every user gets the server-configured value.
  state.idleSec = Math.max(15, parseInt(state.settings.stream_idle_timeout_s, 10) || 30);
  // Remembered voice-button state (localStorage), re-applied when a new live
  // stream reaches "Live" (see applyLiveSound in startPaintCheck).
  loadLiveSoundPref();
  // Bottom-most version label: server APP_VERSION (the HTML text is the fallback).
  const vn = $('#ver-no');
  if (vn && state.settings.app_version) vn.textContent = state.settings.app_version;
  // Same version inside the collapsed nav menu (phones) - one source of truth.
  const vnn = $('#ver-no-nav');
  if (vnn && state.settings.app_version) vnn.textContent = state.settings.app_version;
  loadUsageSelf();     // footer/nav self-view (non-blocking)
  await loadCameras();
  initInstallPrompt(); // PWA install UI (no-op unless Chrome offers it)
  initNotifications(); // notification feed polling + browser alerts + badge
  window.addEventListener('hashchange', onRoute);
  onRoute();
  // Keep the self-view roughly current while the tab is visible (a tiny JSON).
  setInterval(() => { if (!document.hidden) loadUsageSelf(); }, 60000);
}

function onRoute() {
  setNavOpen(false);   // a route change always dismisses the collapsed menu
  const raw = (location.hash || '#/live').replace(/^#\//, '');
  const VIEWS = ['live', 'events', 'fire', 'episodes', 'adaptive', 'scenes',
                 'notifications', 'debug', 'manage', 'user'];
  let view = VIEWS.indexOf(raw) >= 0 ? raw : 'live';
  if (raw === 'usage') view = 'user';   // the Usage tab was removed
  // Debug/Manage are admin-only, and a non-admin who lands on a tab they were
  // not granted falls back to their FIRST allowed tab (or the Account tab).
  if (!viewAllowed(view)) view = USER_VIEWS.find(viewAllowed) || 'user';
  // Live is full-bleed (fills the screen, no dead scroll); other views scroll.
  document.body.classList.toggle('live-full', view === 'live');
  // Events gets its OWN full-bleed layout on a phone held in LANDSCAPE (a
  // "filters | video | clips" 3-column grid - see style.css). The class is
  // harmless in every other layout/view.
  document.body.classList.toggle('events-full', view === 'events');
  if (view !== 'live') stopStream();           // only the Live view streams
  if (view !== 'events') evTeardown();         // release the Events player
  $$('#nav a').forEach(a => a.classList.toggle('active', a.dataset.view === view));
  VIEWS.forEach(v => $('#view-' + v).classList.toggle('hidden', v !== view));
  if (view === 'live') ensureLive();
  else if (view === 'events') loadEvents();
  else if (view === 'fire') reloadFire();   // page-based: reset to page 1 on entry
  else if (view === 'episodes') reloadEpisodes();  // page-based: reset to page 1
  else if (view === 'adaptive') reloadAdaptiveScenes();  // page-based: reset to page 1
  else if (view === 'scenes') reloadScenes();  // page-based: reset to page 1
  else if (view === 'notifications') loadNotifications();  // refresh the feed
  else if (view === 'debug') loadDebug();   // pull the idle sidecar on tab open
  else if (view === 'manage') loadManage(); // admin: users + tab_* permissions
  else if (view === 'user') loadUserView(); // account tab (usage embedded)
}

async function loadCameras() {
  const data = await api('/api/cameras');
  state.cameras = data.cameras || [];
  refreshCamSelects();                 // Events/Fire selects keep a dropdown
  loadEventsPrefs();                   // restore the saved Events camera + class

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
   longer uses a <select> - cameras are picked from the thumbnail row/modal.
   The current choice is PRESERVED when the options are rebuilt (a periodic
   /api/cameras refresh must never silently reset an Events filter). */
function refreshCamSelects() {
  const opts = camNames().map(n => {
    const c = state.cameras.find(x => x.name === n);
    const tag = c && !c.online ? ' (offline)' : '';
    return '<option value="' + esc(n) + '">' + esc(n) + tag + '</option>';
  }).join('');
  const all = '<option value="">all cameras</option>' + opts;
  ['#ev-cam', '#fw-cam', '#sc-cam', '#ep-cam', '#as-cam'].forEach(id => {
    const sel = $(id);
    const prev = sel.value;
    sel.innerHTML = all;
    if (prev && $$('option', sel).some(o => o.value === prev)) sel.value = prev;
  });
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
  /* The live transport could not start, or has ended. Stop the attempt, then
     classify WHY (see decideAfterHlsFail below). Generic over the transport
     (MSE primary or HLS fallback):
     - camera OFFLINE (Frigate online flag false) -> a clean "offline" state,
       no spinner and NO ~1 fps snapshot feed;
     - camera ONLINE but slow to start -> that is LATENCY, so we keep the last
       frozen poster (no snapshot feed) and auto-retry in the background.
       Frigate's online flag (camera_fps) is authoritative, so startup latency
       is never mistaken for an offline camera. */
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || !isLiveTrying()) return;
  clearFrameWatch();                 // no stale first-media watchdog
  clearPaintCheck();
  clearStallWatch();
  destroyMse();                      // tear down the MSE session (WebSocket+MediaSource)
  if (curHls) { try { curHls.destroy(); } catch (e) { /* ignore */ } curHls = null; }
  state.live.imgLive = false;        // freeze the poster; stop the poster poll
  if (liveTimer) { clearTimeout(liveTimer); liveTimer = null; }
  silenceLiveAudio();                // voice must stop with the picture
  const video = $('#live-video');
  try { video.pause(); video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; }
  $('#live-video').classList.add('hidden');
  hideSpinner();
  syncLiveTools();      // no live video: sound off; save/zoom follow the frame
  decideAfterHlsFail(cam, tok);
}

async function decideAfterHlsFail(cam, tok) {
  await fetchCameras();              // fresh Frigate online flags
  if (!state.live.playing || cam !== state.live.cam
      || !isLiveTrying()) return;
  if (!isOnline(cam)) { enterOffline(cam); return; }
  // Camera is ONLINE -> just latency/startup: keep the frozen poster visible,
  // no spinner, no snapshot feed, and auto-retry HLS in the background.
  $('#live-img').classList.remove('hidden');
  syncLiveTools();      // frozen poster: save/zoom active, sound stays off
  $('#live-status').textContent = 'Live starting - auto-retrying\u2026';
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
  destroyMse();
  if (curHls) { try { curHls.destroy(); } catch (e) { /* ignore */ } curHls = null; }
  silenceLiveAudio();                // offline: never keep the voice playing
  const video = $('#live-video');
  try { video.pause(); video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; }
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.add('hidden');
  hideSpinner();
  syncLiveTools();      // offline: no frame/no audio -> controls just disabled
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
  if (retryTimer || !state.live.playing || state.live.imgLive) return;
  if (!MSE_SUPPORTED && !HLS_SUPPORTED) return;   // no transport available
  retryTimer = setTimeout(() => {
    retryTimer = null;
    if (!state.live.playing || state.live.cam !== cam
        || !isLiveTrying() || state.live.imgLive) return;
    $('#live-status').textContent = 'Upgrading to live\u2026';
    startLive(cam, liveTok);   // background attempt; the frozen poster stays up
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
/* Persisted voice-button state (localStorage, like portal.lastCam) so a new
   live stream - or a page reload - re-applies the last choice. */
function loadLiveSoundPref() {
  try {
    const v = localStorage.getItem(LIVE_SOUND_KEY);
    if (v != null) state.live.sound = (v === '1');
  } catch (e) { /* ignore */ }
}
function saveLiveSoundPref() {
  try { localStorage.setItem(LIVE_SOUND_KEY, state.live.sound ? '1' : '0'); }
  catch (e) { /* ignore */ }
}
/* Force the media element silent. Called on EVERY stop/teardown (idle stop,
   camera switch, leaving Live, offline, stream failure) so the voice can never
   outlive the picture. */
function silenceLiveAudio() {
  const video = $('#live-video');
  if (!video) return;
  try { video.muted = true; video.volume = 1; } catch (e) { /* ignore */ }
}
function applyLiveSound() {
  const video = $('#live-video');
  video.muted = !state.live.sound;
  if (!video.muted) video.volume = 1;
  syncSoundUI();
}
function toggleLiveSound() {
  state.live.sound = !state.live.sound;
  saveLiveSoundPref();       // remember the choice for the next stream/session
  applyLiveSound();
  if (state.live.sound) {
    const v = $('#live-video');
    // Re-engage play ONLY if a mute actually paused the element. Re-calling
    // play() on an already-playing live MSE/HLS stream forces a live-edge
    // reset (buffer drop + rebuffer) on go2rtc's shallow ~1 s live window,
    // which itself looks like a stop-then-reconnect.
    if (v && v.paused) { const p = v.play(); if (p && p.catch) p.catch(() => {}); }
    forgiveLiveWatchdogs();   // the unmute decoder pause must not restart HLS
  }
}
/* Unmuting a live stream makes the browser start routing the AAC track, which
   can pause the decoder for a beat while it re-locks A/V sync. On go2rtc's
   shallow live window that hiccup can be misread as a dead/black stream by the
   reveal paint check / post-"Live" stall watchdog, which then DESTROYS and
   reconnects the HLS (the visible "stop the stream, then reconnect with
   voice" loop + possible black screen). Re-baseline whichever live-decode
   watchdog is active so turning sound on is transparent - never a restart. */
function forgiveLiveWatchdogs() {
  if (!state.live.playing || !isLiveTrying() || !state.live.cam) return;
  const cam = state.live.cam, tok = liveTok;
  if (paintCheck) { clearPaintCheck(); startPaintCheck(cam, tok); }
  else if (stallWatch) { armStallWatch(cam, tok); }
}
/* Keep the audible state tied to the <video> element: pausing the stream (idle
   stop, camera switch, leaving Live, teardown) silences the voice and updates
   the button; (re)playing re-applies the remembered choice once live. */
function bindLiveAudioSync() {
  const v = $('#live-video');
  if (!v) return;
  v.addEventListener('pause', () => {
    silenceLiveAudio();       // the voice must never outlive the picture
    syncSoundUI();
  });
  v.addEventListener('play', () => {
    if (state.live.sound && isLiveTrying() && state.live.revealed) applyLiveSound();
    else syncSoundUI();
  });
}
bindLiveAudioSync();
/* Live audio only exists once an HLS video has revealed and really buffered
   media AND is actually playing - during the warm-up poster, an auto-retry
   freeze, an interruption, the offline state or a paused/idle stop there is no
   sound to toggle (a paused element must never read as "sound on"). */
function liveAudioActive() {
  const v = $('#live-video');
  return !!(state.live.playing && isLiveTrying()
            && state.live.revealed && !state.live.imgLive
            && v && !v.classList.contains('hidden') && v.readyState >= 2
            && !v.paused);
}
/* Reflect the REAL sound state (muted until live audio exists). The sound
   button is never hidden - when offline / interrupted / warming up it is just
   DISABLED so the row always shows the same controls. */
function syncSoundUI() {
  const b = $('#live-sound');
  if (!b) return;
  const active = liveAudioActive();
  const on = active && !!state.live.sound;
  b.disabled = !active;
  b.textContent = on ? '\u{1F50A}' : '\u{1F507}';   // 🔊 / 🔇
  b.title = active ? (on ? 'Mute' : 'Enable sound')
                   : 'Sound is available on the live stream';
  b.setAttribute('aria-label', b.title);
  b.classList.toggle('sound-on', on);
}
/* Central refresh for the row's media controls. They stay VISIBLE in every
   state and are merely DISABLED while there is no usable media (offline /
   interrupted / paused): save-image + zoom need a frame on screen, sound needs
   live audio. Call on every stream-state transition and on poster frame load. */
function syncLiveTools() {
  const shot = $('#live-shot');
  if (shot) shot.disabled = !(state.live.cam && zoomable());
  const stop = $('#live-stop');
  if (stop) stop.disabled = !state.live.playing;   // only useful while playing
  syncLiveCenter();     // keep the tap-to-toggle glyph in step with the state
  applyZoom();          // zoom group gated on zoomable() (see below)
  syncSoundUI();
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
  // mode is set by the chosen transport (startMse/startHls) via startLive().
  state.live.imgLive = true;      // poster <img> shown while live warms up
  hideOverlay();
  showSpinner();                  // poster image + spinner until the first media
  scheduleFrame(cam, tok);        // show the freshest detect frame immediately
  startLive(cam, tok);            // ...while live warms up underneath
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
// Watchdog before classifying why the live transport never delivered media. It
// does NOT decide offline by itself - decideAfterHlsFail consults Frigate's
// online flag, so a shorter wait never mislabels slow (latency) cameras.
const HLS_FIRST_FRAME_WAIT_MS = 10000;
function armFirstFrameWatch(cam, tok) {
  if (frameWatchTimer) return;                  // already waiting/done
  frameWatchTimer = setTimeout(() => {          // watchdog: no media in time
    frameWatchTimer = null;
    clearFrameWatch();
    if (state.live.playing && tok === liveTok && cam === state.live.cam
        && isLiveTrying()) hlsFallback(cam, tok);
  }, HLS_FIRST_FRAME_WAIT_MS);
}
function firstMediaReady(cam, tok) {   // first real media buffered -> reveal
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || !isLiveTrying() || state.live.revealed) return;
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
  syncSoundUI();      // sound button stays visible; enabled once live reveals
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

/* ------------- Live view: go2rtc MSE over WebSocket (PRIMARY) ----------------
   The robust live transport - the same one Frigate's own UI uses. One WebSocket
   to the portal (/api/live/<cam>/mse, relayed same-origin to go2rtc's /api/ws).
   The client sends {"type":"mse"}; go2rtc replies with a text frame carrying the
   codec string (e.g. video/mp4; codecs="avc1.4D4029,mp4a.40.2") followed by
   binary fMP4 - an init segment (ftyp/moov) then media fragments (moof/mdat). We
   append those to a MediaSource SourceBuffer and reveal the <video> once the
   first media fragment is buffered. Unlike go2rtc HLS (a shallow ~1 s sliding
   window) this is one continuous stream, so sparse-keyframe (quiet) cameras
   decode reliably. Poster, honest frame-based Live detection and the stall
   watchdog are shared with the HLS fallback above/below. */

function startLive(cam, tok) {
  if (MSE_SUPPORTED) startMse(cam, tok);
  else startHls(cam, tok);
}

function destroyMse() {
  const s = curMse;
  curMse = null;
  if (!s) return;
  try {
    if (s.ws) {
      s.ws.onopen = s.ws.onmessage = s.ws.onerror = s.ws.onclose = null;
      s.ws.close();
    }
  } catch (e) { /* ignore */ }
  try { if (s.buf && s.buf.updating) s.buf.abort(); } catch (e) { /* ignore */ }
  try { if (s.url) URL.revokeObjectURL(s.url); } catch (e) { /* ignore */ }
}

function startMse(cam, tok) {
  state.live.mode = 'mse';
  state.live.revealed = false;
  destroyMse();
  if (curHls) { try { curHls.destroy(); } catch (e) { /* ignore */ } curHls = null; }
  const video = $('#live-video');
  // Keep the <video> visible UNDER the poster (CSS z-index) so it decodes while
  // we wait; the poster drops only once real media is buffered (firstMediaReady).
  video.muted = true;                 // autoplay-safe; sound restored on reveal
  video.classList.remove('hidden');
  syncSoundUI();
  $('#live-status').textContent = 'Connecting ' + cam + '\u2026';
  armFirstFrameWatch(cam, tok);       // watchdog: no media in time -> classify
  const ms = new MediaSource();
  const url = URL.createObjectURL(ms);
  curMse = { ms: ms, url: url, ws: null, buf: null, mime: null,
             queue: [], appended: 0, streaming: false };
  video.src = url;
  ms.addEventListener('sourceopen', function () {
    if (state.live.playing && tok === liveTok && cam === state.live.cam
        && state.live.mode === 'mse') openMseSocket(cam, tok);
  }, { once: true });
  video.addEventListener('loadeddata', function () {
    firstMediaReady(cam, tok);
  }, { once: true });
}

function mseOk(cam, tok) {
  return !!(state.live.playing && tok === liveTok && cam === state.live.cam
            && state.live.mode === 'mse' && curMse);
}

function openMseSocket(cam, tok) {
  const proto = (location.protocol === 'https:') ? 'wss://' : 'ws://';
  const url = proto + location.host + '/api/live/'
    + encodeURIComponent(cam) + '/mse';
  let ws;
  try { ws = new WebSocket(url); } catch (e) { hlsFallback(cam, tok); return; }
  ws.binaryType = 'arraybuffer';
  if (!curMse) { try { ws.close(); } catch (e) { /* ignore */ } return; }
  curMse.ws = ws;
  ws.onopen = function () {
    if (!mseOk(cam, tok)) { try { ws.close(); } catch (e) { /* ignore */ } return; }
    ws.send(JSON.stringify({ type: 'mse' }));   // go2rtc: enter MSE mode
  };
  ws.onmessage = function (ev) {
    if (!mseOk(cam, tok)) return;
    if (typeof ev.data === 'string') { mseControl(cam, tok, ev.data); return; }
    if (curMse.buf) {
      curMse.queue.push(new Uint8Array(ev.data));
      pumpMse(cam, tok);
    }
  };
  ws.onerror = function () { /* onclose follows */ };
  ws.onclose = function () { if (mseOk(cam, tok)) hlsFallback(cam, tok); };
}

/* go2rtc MSE control text: {"type":"mse","value":"<mime codecs>"}. Create the
   SourceBuffer from that mime; if this browser cannot play the codecs, fall
   back to HLS (e.g. native Safari) rather than showing a dead stream. */
function mseControl(cam, tok, text) {
  let m = null;
  try { m = JSON.parse(text); } catch (e) { return; }
  if (!m || m.type !== 'mse' || !m.value || !curMse || curMse.buf) return;
  if (!window.MediaSource || !MediaSource.isTypeSupported(m.value)) {
    destroyMse();
    if (state.live.playing && tok === liveTok && cam === state.live.cam) {
      if (HLS_SUPPORTED) startHls(cam, tok); else hlsFallback(cam, tok);
    }
    return;
  }
  curMse.mime = m.value;
  try {
    const buf = curMse.ms.addSourceBuffer(m.value);
    buf.mode = 'segments';
    buf.addEventListener('updateend', function () {
      trimMse();
      pumpMse(cam, tok);
    });
    buf.addEventListener('error', function () { hlsFallback(cam, tok); });
    curMse.buf = buf;
  } catch (e) {
    hlsFallback(cam, tok);
  }
}

/* Append queued fMP4 chunks one at a time (a SourceBuffer allows a single
   in-flight append). Reveal the <video> once a media fragment (moof) is
   buffered - the init segment (ftyp/moov) alone carries no frames. */
function pumpMse(cam, tok) {
  const s = curMse;
  if (!s || !s.buf || !s.ms || s.ms.readyState !== 'open') return;
  if (s.buf.updating || !s.queue.length || !mseOk(cam, tok)) return;
  const chunk = s.queue.shift();
  let hasMedia = false;
  for (let i = 0; i + 3 < chunk.length; i++) {
    if (chunk[i] === 0x6d && chunk[i + 1] === 0x6f &&
        chunk[i + 2] === 0x6f && chunk[i + 3] === 0x66) { hasMedia = true; break; }
  }
  try {
    s.buf.appendBuffer(chunk);
  } catch (e) {
    // Buffer full/invalid: drop some history, then retry on the next updateend.
    try {
      if (s.buf.buffered.length) {
        s.buf.remove(0, s.buf.buffered.start(0) + MSE_BACK_BUFFER_S);
      }
    } catch (e2) { /* ignore */ }
    s.queue.unshift(chunk);
    return;
  }
  s.appended += 1;
  if (hasMedia && !s.streaming) {
    s.streaming = true;
    const v = $('#live-video');
    const p = (v && v.play()) || null;
    if (p && p.catch) p.catch(function () { /* ignore */ });
    firstMediaReady(cam, tok);     // reveal the video + start the paint check
  }
}

/* Keep only a short back-buffer behind the playhead so a long watch does not
   grow memory without bound. */
function trimMse() {
  const s = curMse;
  if (!s || !s.buf || s.buf.updating) return;
  try {
    const v = $('#live-video');
    const ct = v ? v.currentTime : 0;
    if (s.buf.buffered.length &&
        ct - s.buf.buffered.start(0) > MSE_BACK_BUFFER_S + 5) {
      s.buf.remove(0, ct - MSE_BACK_BUFFER_S);
    }
  } catch (e) { /* ignore */ }
}

/* A real video data fragment has buffered -> reveal the <video> (drop the
   poster). "Live (HLS)" is only claimed once a frame actually renders
   (startPaintCheck) - otherwise we keep the poster and auto-retry instead of
   a black screen that says Live. */
function firstHlsFrame(cam, tok) {
  if (!state.live.playing || tok !== liveTok || cam !== state.live.cam
      || !isLiveTrying()) return;
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
  syncLiveTools();      // real frame now live: save/zoom + sound become active
  $('#live-status').textContent = 'Starting\u2026';
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
        || !isLiveTrying() || state.live.imgLive) {
      clearPaintCheck(); return;
    }
    const fr = canCount ? videoFrames(video) : -1;
    // painted = a new video frame was actually decoded (not just audio).
    const painted = canCount ? fr > fr0 : video.currentTime !== lastT;
    lastT = video.currentTime;
    if (painted && video.readyState >= 2) {
      clearPaintCheck();
      $('#live-status').textContent = 'Live';
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
        || !isLiveTrying()) { clearStallWatch(); return; }
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
      || !isLiveTrying()) return;
  clearPaintCheck();
  clearStallWatch();
  state.live.revealed = true;         // this attempt already revealed once
  state.live.imgLive = false;         // frozen poster (no ~1 fps feed)
  destroyMse();                       // tear down the MSE session (if any)
  if (curHls) { try { curHls.destroy(); } catch (e) { /* ignore */ } curHls = null; }
  silenceLiveAudio();                // black revert: stop the voice too
  const video = $('#live-video');
  try { video.pause(); video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; }
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.remove('hidden');
  hideSpinner();
  syncLiveTools();      // frozen poster still zoomable/savable; sound off
  $('#live-status').textContent = 'Live starting - waiting for video\u2026';
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
    syncLiveTools();   // poster frame arrived -> save/zoom become usable
  };
  img.onerror = () => {
    if (state.live.playing && cam === state.live.cam && tok === liveTok
        && state.live.imgLive) {
      liveTimer = setTimeout(() => scheduleFrame(cam, tok), 3000);
    }
    syncLiveTools();   // no poster frame -> controls follow what is on screen
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
  destroyMse();
  if (curHls) { curHls.destroy(); curHls = null; }
  silenceLiveAudio();                // idle stop / leave: voice stops with video
  const video = $('#live-video');
  try { video.pause(); } catch (e) { /* ignore */ }
  try { video.removeAttribute('src'); video.load(); } catch (e) { /* ignore */ }
  const img = $('#live-img');
  if (img) { img.onload = null; img.onerror = null; img.removeAttribute('src'); }
  $('#live-video').classList.add('hidden');
  $('#live-img').classList.add('hidden');
  hideSpinner();
  syncLiveTools();      // stopped: no frame/no audio -> controls just disabled
  $('#live-status').textContent = '';
  resetZoom();                 // a fresh camera / restart starts at 1x, no stale pan
}

function resume() {
  if (!state.live.cam) return;
  startStream(state.live.cam);
}

/* Stop the live stream on demand (the #live-stop button): tear it down and
   offer Resume via the shared overlay - the same UX as the idle stop. */
function stopLive() {
  if (!state.live.cam) return;
  stopStream();
  showOverlay('Live stopped.');
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
  // Zoom needs a real frame on screen: in offline/interrupted/paused states the
  // whole group is DISABLED (still visible, just inactive).
  const frame = zoomable();
  const zi = $('#zoom-in'), zo = $('#zoom-out'), zr = $('#zoom-reset');
  if (zi) zi.disabled = !frame || zoom.s >= ZOOM_MAX;
  if (zo) zo.disabled = !frame || zoom.s <= ZOOM_MIN;
  if (zr) zr.disabled = !frame || zoom.s <= ZOOM_MIN;
  const st = liveStage();
  if (st) st.classList.toggle('zoomed', frame && zoom.s > ZOOM_MIN);
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

/* Tap the live frame -> a YouTube-style center play/stop button appears; tapping
   it stops the stream (or resumes it) and hides. A pan DRAG is not a tap (we use
   a small movement threshold so zoom-panning never toggles it). */
let liveTapMoved = false, liveTapX = 0, liveTapY = 0;
function syncLiveCenter() {
  const b = $('#live-center');
  if (!b) return;
  const playing = !!state.live.playing;
  b.textContent = playing ? '\u23F9' : '\u25B6';   // \u23F9 stop / \u25B6 play
  b.title = playing ? 'Stop' : 'Play';
  b.setAttribute('aria-label', b.title);
}
function toggleLiveCenter() {
  const b = $('#live-center');
  if (!b) return;
  if (b.classList.contains('hidden')) { syncLiveCenter(); b.classList.remove('hidden'); }
  else b.classList.add('hidden');
}
if (stageEl) {
  stageEl.addEventListener('pointerdown', (e) => {
    liveTapX = e.clientX; liveTapY = e.clientY; liveTapMoved = false;
  });
  stageEl.addEventListener('pointermove', (e) => {
    if (Math.abs(e.clientX - liveTapX) > 8 ||
        Math.abs(e.clientY - liveTapY) > 8) liveTapMoved = true;
  });
  stageEl.addEventListener('click', (e) => {
    if (liveTapMoved) return;                    // it was a pan drag
    if (e.target.closest('button, .overlay, .spinner')) return;
    toggleLiveCenter();
  });
}
$('#live-center').addEventListener('click', (e) => {
  e.stopPropagation();
  if (state.live.playing) stopLive(); else resume();
  $('#live-center').classList.add('hidden');
});
syncLiveTools();   // initial state: no frame/no audio yet -> row tools disabled

/* -------- camera switching: thumbnail row / modal ------------------------- */
function camNames() {
  return state.cameras.map(c => c.name).filter(n => n);
}
function rememberCam(cam) {
  state.live.cam = cam;                 // startStream also sets it (idempotent)
  try { localStorage.setItem('portal.lastCam', cam); } catch (e) { /* ignore */ }
}
/* Pick a camera from the thumbnail row / modal. Clicking the camera that is
   already streaming just closes the picker (no restart). */
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
// Stop the live stream on demand (offers Resume through the overlay).
$('#live-stop').addEventListener('click', stopLive);
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
// `defaultKey` (optional) selects an initial preset - the Events tab passes '24h'
// so it opens on the last day instead of "All time"; Fire/Scenes keep the default.
function wireTimeControls(prefix, onApply, defaultKey) {
  const timeSel = $('#' + prefix + '-time');
  const wrap = $('#' + prefix + '-custom-wrap');
  const fromEl = $('#' + prefix + '-from');
  const toEl = $('#' + prefix + '-to');
  fillTimeSelect(timeSel);
  if (defaultKey) timeSel.value = defaultKey;
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

/* ---------------- Frigate events: large player + clip strip + playlist --------
   The filter bar (Camera/Class/Time/Refresh) is UNCHANGED; it feeds ONE large
   <video> (#ev-video) through a horizontally scrolling strip of clip buttons
   (#ev-strip). Replaces the old grid of per-card inline videos.
     - evAll      : the filtered events (newest first) = the playlist order.
     - evSelIdx   : index into evAll of the clip in the stage (-1 = none).
     - evPlaylist : the auto-advance switch (ON -> play the next clip on 'ended').
     - idle 60 s  : with no user interaction, turn the playlist OFF + pause and
                    show the Resume overlay (never auto-play the next clip).
   Clips stream from /api/events/<id>/clip.mp4 (seekable - the portal proxy now
   forwards Range); the poster is /api/events/<id>/snapshot.jpg.
   See plans/portal-events-tab-player.md. */
const EV_PAGE = 24;
const EV_MAX = 5000;        // client-side cap for a single Frigate fetch
const EV_IDLE_MS = 60000;   // idle stop: 1 minute (fixed; Live uses STREAM_IDLE_TIMEOUT_S)
const EV_STORE = { cam: 'portal.evCam', label: 'portal.evLabel' };

let evAll = [];             // current filter's full event array (newest first)
let evShown = 0;            // how many clips are in the strip (grows on Load more)
let evSelIdx = -1;          // index into evAll of the clip in the stage
let evPlaylist = false;     // auto-advance switch
let evPlaylistResume = false;  // playlist was ON when the idle stop fired
let evIdleTimer = null;

/* ---- remembered filters: last camera + last class (localStorage, best effort) */
function loadEventsPrefs() {
  try {
    const cam = localStorage.getItem(EV_STORE.cam);
    const sel = $('#ev-cam');
    if (cam != null && sel && $$('option', sel).some(o => o.value === cam)) {
      sel.value = cam;
    }
    const lab = localStorage.getItem(EV_STORE.label);
    if (lab != null) $('#ev-label').value = lab;
  } catch (e) { /* localStorage unavailable - filters just stay at defaults */ }
}
function saveEvCam() {
  try { localStorage.setItem(EV_STORE.cam, $('#ev-cam').value); } catch (e) { /* ignore */ }
}
function saveEvLabel() {
  try { localStorage.setItem(EV_STORE.label, $('#ev-label').value.trim()); } catch (e) { /* ignore */ }
}

/* ---- idle stop: 60 s without USER interaction stops the playlist ----
   The timer is (re)armed only by real interaction (and by enabling the
   playlist / resuming), NEVER by an auto-advance - otherwise a running
   playlist would keep resetting it and the stop would never fire. */
function armEvIdle() {
  clearEvIdle();
  if (evSelIdx < 0) return;
  evIdleTimer = setTimeout(onEvIdle, EV_IDLE_MS);
}
function clearEvIdle() {
  if (evIdleTimer) { clearTimeout(evIdleTimer); evIdleTimer = null; }
}
function onEvIdle() {
  evIdleTimer = null;
  if (evSelIdx < 0) return;
  if (!evPlaylist) return;              // nothing to stop - leave the clip alone
  evPlaylistResume = true;              // Resume restores this
  evPlaylist = false;
  syncEvPlaylist();
  const v = $('#ev-video');
  if (v && !v.paused) v.pause();        // never fetch/play the next clip
  showEvOverlay('Playlist paused after 1 minute of inactivity.');
}
function showEvOverlay(msg) {
  $('#ev-overlay-msg').textContent = msg;
  $('#ev-overlay').classList.remove('hidden');
}
function hideEvOverlay() { $('#ev-overlay').classList.add('hidden'); }

/* Resume (user gesture): continue the clip and restore the playlist it had. */
function resumeEvPlayback() {
  hideEvOverlay();
  if (evPlaylistResume) { evPlaylist = true; evPlaylistResume = false; syncEvPlaylist(); }
  const v = $('#ev-video');
  if (v && v.currentSrc) { const p = v.play(); if (p && p.catch) p.catch(() => {}); }
  armEvIdle();
}

/* ---- playlist switch ---- */
function syncEvPlaylist() {
  const b = $('#ev-playlist');
  if (!b) return;
  const ev = evAll[evSelIdx];
  b.disabled = !(ev && ev.has_clip);
  b.setAttribute('aria-pressed', evPlaylist ? 'true' : 'false');
  b.textContent = 'Playlist: ' + (evPlaylist ? 'on' : 'off');
}
function toggleEvPlaylist() {
  if (evSelIdx < 0) return;
  evPlaylist = !evPlaylist;
  evPlaylistResume = false;
  hideEvOverlay();
  syncEvPlaylist();
  armEvIdle();
  const v = $('#ev-video');
  if (evPlaylist && v && v.ended) evAdvance();   // already finished -> next now
}
function nextPlayableIdx(from) {
  for (let i = from + 1; i < evAll.length; i++) {
    if (evAll[i] && evAll[i].has_clip) return i;
  }
  return -1;
}
function evAdvance() {
  const next = nextPlayableIdx(evSelIdx);
  if (next < 0) {                       // end of the playlist: stop, no loop
    evPlaylist = false;
    syncEvPlaylist();
    $('#ev-status').textContent = 'End of playlist.';
    return;
  }
  evSelect(next, true);
}

/* ---- playback ---- */
function evClipUrl(ev) { return '/api/events/' + encodeURIComponent(ev.id) + '/clip.mp4'; }
function evSnapUrl(ev) { return '/api/events/' + encodeURIComponent(ev.id) + '/snapshot.jpg'; }

// Show the event at `idx` in the large frame; `autoplay` starts it immediately.
function evSelect(idx, autoplay) {
  if (idx < 0 || idx >= evAll.length) return;
  evSelIdx = idx;
  const ev = evAll[idx];
  const v = $('#ev-video');
  const stage = $('#ev-stage');
  hideEvOverlay();
  hideEvCenter();
  if (stage) stage.classList.add('has-clip');
  if (v) {
    try { v.pause(); } catch (e) { /* ignore */ }
    if (ev.has_clip) {
      v.poster = evSnapUrl(ev);
      v.src = evClipUrl(ev);
      try { v.currentTime = 0; } catch (e) { /* ignore */ }
      if (autoplay) { const p = v.play(); if (p && p.catch) p.catch(() => {}); }
    } else {
      // Snapshot only: no clip to play, just the still in the frame.
      try { v.removeAttribute('src'); v.load(); } catch (e) { /* ignore */ }
      v.poster = evSnapUrl(ev);
    }
  }
  if (!ev.has_clip) evPlaylist = false;   // nothing to auto-advance from a still
  syncEvPlaylist();
  renderEvNow();
  if (idx >= evShown) {
    // Selected beyond the rendered window (e.g. playlist advance): grow it.
    evShown = Math.min(evAll.length, Math.ceil((idx + 1) / EV_PAGE) * EV_PAGE);
    renderEvClips();
  } else {
    markEvActive();
  }
}
function renderEvNow() {
  const el = $('#ev-now');
  if (!el) return;
  const ev = evAll[evSelIdx];
  if (!ev) { el.textContent = ''; return; }
  const parts = [ev.camera || '', ev.label || 'detection', fmtDT(ev.start_time)];
  if (!ev.has_clip) parts.push('(snapshot only)');
  el.textContent = parts.filter(Boolean).join('  \u00b7  ');
}
function markEvActive() {
  const strip = $('#ev-strip');
  if (!strip) return;
  $$('.ev-clip', strip).forEach(b => {
    const on = Number(b.dataset.idx) === evSelIdx;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
    if (on) b.scrollIntoView({ block: 'nearest', inline: 'center' });
  });
}

// Stop the player and drop the selection (leaving the tab / changing filters).
function evTeardown() {
  clearEvIdle();
  evExitFullscreen();          // leaving the player: drop any stage fullscreen
  hideEvOverlay();
  hideEvCenter();
  evPlaylist = false;
  evPlaylistResume = false;
  evSelIdx = -1;
  if (evClipsAuto) { evClipsAuto = false; setEvClipsCollapsed(false); }
  const v = $('#ev-video');
  if (v) {
    try { v.pause(); v.removeAttribute('src'); v.load(); } catch (e) { /* ignore */ }
    v.poster = '';
  }
  const stage = $('#ev-stage');
  if (stage) stage.classList.remove('has-clip');
  syncEvPlaylist();
  syncEvFs();
  renderEvNow();
}

$('#ev-refresh').addEventListener('click', loadEvents);
$('#ev-cam').addEventListener('change', () => { saveEvCam(); loadEvents(); });
$('#ev-label').addEventListener('change', () => { saveEvLabel(); loadEvents(); });
wireTimeControls('ev', loadEvents, '24h');   // default = Last 24 hours
$('#ev-playlist').addEventListener('click', toggleEvPlaylist);
$('#ev-overlay-resume').addEventListener('click', resumeEvPlayback);

/* ---- custom FULLSCREEN on the STAGE (never the <video> element) ----
   The browser's native fullscreen targets the <video> element, which puts the
   idle-stop Resume overlay (#ev-overlay - its SIBLING inside #ev-stage) OUTSIDE
   the fullscreen view: an idle-stopped playlist then just looks frozen and the
   user never sees the Resume button. The native button is suppressed
   (controlslist="nofullscreen" on #ev-video) and #ev-fs fullscreens the whole
   STAGE instead, so the caption bar, the center button AND the Resume overlay
   all stay visible in fullscreen. */
function evFsElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}
function evStageFullscreen() { return evFsElement() === $('#ev-stage'); }
function syncEvFs() {
  const b = $('#ev-fs');
  if (!b) return;
  const on = evStageFullscreen();
  b.textContent = on ? '\u2715' : '\u26F6';   // \u2715 exit / \u26F6 enter
  b.title = on ? 'Exit fullscreen' : 'Fullscreen';
  b.setAttribute('aria-label', b.title);
  b.setAttribute('aria-pressed', on ? 'true' : 'false');
}
function evExitFullscreen() {
  if (!evFsElement()) return;
  const ex = document.exitFullscreen || document.webkitExitFullscreen;
  if (!ex) return;
  try { const p = ex.call(document); if (p && p.catch) p.catch(() => {}); } catch (e) { /* ignore */ }
}
function evToggleFullscreen() {
  const stage = $('#ev-stage');
  if (!stage) return;
  if (evFsElement()) { evExitFullscreen(); return; }
  const req = stage.requestFullscreen || stage.webkitRequestFullscreen;
  if (!req) return;   // no Fullscreen API: leave the native controls to it
  try { const p = req.call(stage); if (p && p.catch) p.catch(() => {}); } catch (e) { /* ignore */ }
}
$('#ev-fs').addEventListener('click', evToggleFullscreen);
// Keep the button in sync, and bail OUT of any stray <video> fullscreen (a
// browser without controlslist, or a double-click Chrome maps to video
// fullscreen) so the Resume overlay can never end up hidden behind it.
['fullscreenchange', 'webkitfullscreenchange'].forEach(evt =>
  document.addEventListener(evt, () => {
    if (evFsElement() === $('#ev-video')) evExitFullscreen();
    syncEvFs();
  }));
syncEvFs();
// Collapse / expand the clip THUMBNAILS (the named "Clips" hamburger). Shown on
// phones in BOTH orientations: collapsing hides the strip so the frame gets the
// vertical space; in landscape it also shrinks the column to a slim handle.
// Hidden on tablet/desktop (CSS), where the strip is always visible.
let evClipsAuto = false;   // true while WE collapsed it for playback (theater)
function setEvClipsCollapsed(collapsed) {
  document.body.classList.toggle('ev-clips-collapsed', !!collapsed);
  const t = $('#ev-clips-toggle');
  if (t) t.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
}
// The toggle only exists in the phone layouts; offsetParent is null when the
// button is display:none (tablet/desktop), so auto-collapse stays scoped there.
function evClipsToggleVisible() {
  const t = $('#ev-clips-toggle');
  return !!t && t.offsetParent !== null;
}
$('#ev-clips-toggle').addEventListener('click', () => {
  evClipsAuto = false;   // a manual toggle wins until the next play
  setEvClipsCollapsed(!document.body.classList.contains('ev-clips-collapsed'));
});
// Theater behaviour (all layouts with the toggle): when a clip PLAYS, collapse
// the thumbnails; restore them when it STOPS (pause / ended). Only if WE
// collapsed it - a manual choice is respected.
$('#ev-video').addEventListener('play', () => {
  if (!evClipsToggleVisible()) return;
  if (!document.body.classList.contains('ev-clips-collapsed')) {
    evClipsAuto = true;
    setEvClipsCollapsed(true);
  }
});
['pause', 'ended'].forEach(evt => $('#ev-video').addEventListener(evt, () => {
  if (evClipsAuto) { evClipsAuto = false; setEvClipsCollapsed(false); }
}));
// Playlist auto-advance: the current clip ended -> play the next one.
$('#ev-video').addEventListener('ended', () => { if (evPlaylist) evAdvance(); });
// Any interaction inside the tab is a user present: (re)arm the idle stop.
['pointerdown', 'keydown', 'wheel'].forEach(evt =>
  $('#view-events').addEventListener(evt, armEvIdle, { passive: true }));
// Click a clip in the strip: select + play it in the large frame. The trailing
// "Load more" button appends the next window of clips - the SAME behaviour
// whether the strip is horizontal (phones/tablet) or vertical (phone landscape).
$('#ev-strip').addEventListener('click', (e) => {
  if (e.target.closest('#ev-loadmore')) { loadMoreEvClips(); return; }
  const btn = e.target.closest('.ev-clip');
  if (!btn) return;
  evPlaylistResume = false;
  evSelect(Number(btn.dataset.idx), true);
});

/* Tap the events frame -> a YouTube-style center play/pause button; tapping the
   button toggles playback. Ignored when nothing is loaded and on taps in the
   native controls strip (bottom ~48px). */
function evPlayingNow() {
  const v = $('#ev-video');
  return !!(v && v.currentSrc && !v.paused && !v.ended);
}
function syncEvCenter() {
  const b = $('#ev-center');
  if (!b) return;
  const playing = evPlayingNow();
  b.textContent = playing ? '\u23F8' : '\u25B6';   // \u23F8 pause / \u25B6 play
  b.title = playing ? 'Pause' : 'Play';
  b.setAttribute('aria-label', b.title);
}
function hideEvCenter() {
  const b = $('#ev-center');
  if (b) b.classList.add('hidden');
}
function toggleEvCenter() {
  const b = $('#ev-center');
  if (!b) return;
  if (b.classList.contains('hidden')) { syncEvCenter(); b.classList.remove('hidden'); }
  else b.classList.add('hidden');
}
$('#ev-center').addEventListener('click', (e) => {
  e.stopPropagation();
  const v = $('#ev-video');
  if (!v || !v.currentSrc) return;
  if (v.paused) { const p = v.play(); if (p && p.catch) p.catch(() => {}); }
  else v.pause();
  hideEvCenter();
});
$('#ev-stage').addEventListener('click', (e) => {
  if (e.target.closest('button, .ev-now-bar, .ev-overlay')) return;
  const v = $('#ev-video');
  if (!v || !v.currentSrc) return;                 // nothing loaded to control
  const stage = $('#ev-stage');
  const r = stage.getBoundingClientRect();
  if (e.clientY > r.bottom - 48) return;           // native controls strip
  toggleEvCenter();
});
['play', 'pause', 'ended'].forEach(evt => $('#ev-video').addEventListener(evt, () => {
  const b = $('#ev-center');
  if (b && !b.classList.contains('hidden')) syncEvCenter();
}));

async function loadEvents() {
  const st = $('#ev-status');
  st.textContent = 'Loading…';
  try {
    const { after, before } = timeRange($('#ev-time'), $('#ev-from'), $('#ev-to'));
    const p = new URLSearchParams({ limit: String(EV_MAX) });
    const cam = $('#ev-cam').value; if (cam) p.set('camera', cam);
    const lab = $('#ev-label').value.trim(); if (lab) p.set('label', lab);
    if (after) p.set('after', String(after));
    if (before) p.set('before', String(before));
    const events = await api('/api/events?' + p.toString());
    evAll = Array.isArray(events) ? events : [];
    evShown = Math.min(evAll.length, EV_PAGE);
    evTeardown();                     // new filter: drop the old clip + playlist
    st.textContent = '';
    renderEvClips();
    // Ready the frame with the first PLAYABLE clip's poster (no autoplay): the
    // stage is populated without spending bandwidth until the user presses play.
    const first = evAll.findIndex(x => x && x.has_clip);
    if (first >= 0) evSelect(first, false);
    else if (evAll.length) evSelect(0, false);   // snapshots only
  } catch (e) {
    evAll = [];
    evShown = 0;
    evTeardown();
    st.textContent = '';
    const cnt = $('#ev-count'); if (cnt) cnt.textContent = '';
    $('#ev-strip').innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
  }
}

// Render the clip list: the first evShown events plus a trailing control -
// "Load more" while events remain, or "No more videos" once all are shown.
// evShown grows by EV_PAGE per click; ONE rendering serves both the horizontal
// strip (phones/tablet) and the vertical strip (phone landscape).
function renderEvClips() {
  if (evShown > evAll.length) evShown = evAll.length;
  if (evShown <= 0) evShown = Math.min(evAll.length, EV_PAGE);
  const cnt = $('#ev-count');
  if (cnt) {
    cnt.textContent = evAll.length
      ? evAll.length + ' event(s)' +
        (evAll.length >= EV_MAX ? ' (older omitted)' : '')
      : '';
  }
  renderEvStrip(evAll.slice(0, evShown), 0);
}

// One clip button (shared by the first render and the "Load more" append).
function evClipHtml(ev, idx) {
  const glyph = ev.has_clip ? '<span class="ev-playglyph">&#9654;</span>' : '';
  const thumb = ev.has_snapshot
    ? '<img loading="lazy" src="' + evSnapUrl(ev) + '" alt="">'
    : '<span class="ev-noclip">no preview</span>';
  return '<button class="ev-clip" type="button" role="option" data-idx="' + idx +
    '" aria-selected="false" title="' + esc(fmtDT(ev.start_time)) + '">' +
    '<span class="ev-thumb">' + thumb + glyph + '</span>' +
    '<span class="ev-meta">' +
      '<span class="ev-lab">' + esc(ev.label || 'detection') + '</span>' +
      '<span class="ev-time">' + esc(ev.camera || '') + ' &middot; ' +
        esc(fmtDT(ev.start_time)) + '</span>' +
    '</span>' +
  '</button>';
}
// The trailing control: "Load more" while events remain, else "No more videos"
// (uniform for the horizontal strip and the vertical landscape strip).
function evTailHtml() {
  return evShown < evAll.length
    ? '<button id="ev-loadmore" class="ev-loadmore" type="button">Load more</button>'
    : '<div class="ev-end">No more videos</div>';
}
// The lower navigation: one button per clip, then the trailing control.
function renderEvStrip(events, baseIdx) {
  const box = $('#ev-strip');
  if (!Array.isArray(events) || !events.length) {
    box.innerHTML = '<div class="empty">No events</div>';
    return;
  }
  box.innerHTML = events.map((ev, i) => evClipHtml(ev, baseIdx + i)).join('') +
    evTailHtml();
  markEvActive();
}
// "Load more": APPEND the next window (preserving the scroll position) by
// replacing the old trailing control with the new clips + a fresh control. The
// already-active clip is left untouched (so nothing scrolls out from under the
// user - do NOT call markEvActive here).
function loadMoreEvClips() {
  if (evShown >= evAll.length) return;
  const box = $('#ev-strip');
  const from = evShown;
  evShown = Math.min(evAll.length, evShown + EV_PAGE);
  const html = evAll.slice(from, evShown)
    .map((ev, i) => evClipHtml(ev, from + i)).join('') + evTailHtml();
  const tail = $('#ev-loadmore', box) || $('.ev-end', box);
  if (tail) tail.outerHTML = html;
  else box.insertAdjacentHTML('beforeend', html);
}

/* ---------------- Firewatch fire detections & alerts (paged + time-filtered) -- */
const FW_PAGE = 24;
let fwPage = 1;
let fwPages = 1;

function reloadFire() {
  fwPage = 1;
  fwDets = {};
  fwMeta = {};
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

/* Derived evidence tags (inferred portal-side, see portal/firestore.py):
   `motion` = the sample's winning box sat near motion (the firewatch motion
   bonus applied); `hits` = how many consecutive stored evidence frames this
   camera produced in the same burst (a proxy for the alert's confirm count). */
function fireMetaTags(f) {
  const hits = Number(f.hits) || 1;
  return (f.motion
      ? '<span class="tag motion" title="Corroborated by motion (firewatch motion bonus)">motion</span>'
      : '') +
    '<span class="tag hits" title="Consecutive stored evidence frames for this camera in this burst">' +
      hits + ' hit' + (hits === 1 ? '' : 's') + '</span>';
}

function fireCard(f) {
  fwDets[f.id] = f.detections || [];
  fwMeta[f.id] = f;
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
      fireMetaTags(f) +
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

/* -------- fire evidence lightbox: click a card image to view it larger ----- */
function openFireLightbox(id) {
  const f = fwMeta[id];
  if (!f) return;
  const img = $('#fw-lb-img');
  const labels = [...new Set((f.detections || []).map(d => d.label))];
  const labelTags = labels.map(l => {
    const c = String(l).toLowerCase();
    return '<span class="tag ' + (c === 'smoke' ? 'smoke' : 'fire') + '">' + esc(l) + '</span>';
  }).join(' ');
  $('#fw-lb-title').textContent = 'Fire evidence #' + id;
  $('#fw-lb-meta').innerHTML =
    (f.alerted ? '<span class="tag alerted">alerted</span>'
               : '<span class="tag fire">detection</span>') +
    labelTags +
    fireMetaTags(f) +
    '<span>best ' + Number(f.best_score).toFixed(2) + '</span>' +
    '<span class="tag">' + esc(f.camera || '') + '</span>' +
    '<span class="muted">' + esc(f.ts_utc || '') + '</span>';
  // Clear, then load the (cached) image; the detection/smoke boxes are drawn
  // once the natural size is known so the percentages map onto the full image.
  $('#fw-lb-boxes').innerHTML = '';
  img.onload = () => drawFireLightboxBoxes(img, id);
  img.onerror = () => { img.onload = null; img.onerror = null; };
  $('#fw-lightbox').classList.remove('hidden');
  document.body.classList.add('lb-open');
  img.src = '/api/fire/' + id + '/image.jpg';
  if (img.complete) drawFireLightboxBoxes(img, id);   // served from cache
}

function drawFireLightboxBoxes(img, id) {
  const boxes = $('#fw-lb-boxes');
  if (!boxes || !id) return;
  boxes.innerHTML = '';
  if (!img.complete || !img.naturalWidth) return;
  const nw = img.naturalWidth || 640;
  const nh = img.naturalHeight || 360;
  boxes.innerHTML = (fwDets[id] || []).map(d => {
    const x1 = d.x1 / nw * 100, y1 = d.y1 / nh * 100;
    const x2 = d.x2 / nw * 100, y2 = d.y2 / nh * 100;
    const cls = String(d.label).toLowerCase() === 'smoke' ? 'smoke' : '';
    return '<div class="fw-box ' + cls + '" style="left:' + x1 + '%;top:' + y1 + '%;' +
      'width:' + Math.max(0, x2 - x1) + '%;height:' + Math.max(0, y2 - y1) + '%"></div>';
  }).join('');
}

function closeFireLightbox() {
  const lb = $('#fw-lightbox');
  if (!lb || lb.classList.contains('hidden')) return;
  lb.classList.add('hidden');
  document.body.classList.remove('lb-open');
  const img = $('#fw-lb-img');
  img.onload = null;
  img.onerror = null;
  img.removeAttribute('src');
  $('#fw-lb-boxes').innerHTML = '';
}

// Fire card images open the lightbox (delegated so it survives every re-render;
// clicks reach the <img> because the .fw-overlays layer is pointer-events:none).
$('#fw-list').addEventListener('click', (e) => {
  const img = e.target.closest('.fw-img');
  if (img && img.dataset.id) openFireLightbox(img.dataset.id);
});
$('#fw-lb-close').addEventListener('click', closeFireLightbox);
$('#fw-lightbox').addEventListener('click', (e) => {
  if (e.target.closest('[data-fw-close]')) closeFireLightbox();   // backdrop tap
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeFireLightbox();
});

/* ------------- scenereader EPISODES (cross-camera person stories) ----------
   One card per episode: the narrative in English AND the same narrative in
   Arabic. Both are composed DETERMINISTICALLY by scenereader from the same
   facts (no model), so the two languages can never disagree. Underneath are the
   ordered supporting captures, served by the EXISTING Frigate snapshot proxy.
   "Process now" only ASKS the service for a caption batch; the service still
   applies its idle governor, so the button can never force a hot host to work. */
const EP_LIMIT = 20;
let epPage = 1;
let epPages = 1;

function epClock(epoch) {
  if (!epoch) return '';
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function reloadEpisodes() {
  epPage = 1;
  loadEpisodes();
}
$('#ep-refresh').addEventListener('click', reloadEpisodes);
$('#ep-cam').addEventListener('change', reloadEpisodes);
$('#ep-day').addEventListener('change', reloadEpisodes);
$('#ep-prev').addEventListener('click', () => {
  if (epPage > 1) { epPage--; loadEpisodes(); }
});
$('#ep-next').addEventListener('click', () => {
  if (epPage < epPages) { epPage++; loadEpisodes(); }
});
$('#ep-drain').addEventListener('click', async () => {
  const st = $('#ep-status');
  const btn = $('#ep-drain');
  btn.disabled = true;
  try {
    // Plain fetch (POST + cookie) so this does not depend on the api() helper's
    // option signature.
    const r = await fetch('/api/scenereader/drain', {
      method: 'POST', credentials: 'same-origin',
    });
    st.textContent = r.ok
      ? 'Caption batch requested - it runs as soon as the host is idle.'
      : 'Could not request a batch (HTTP ' + r.status + ').';
  } catch (err) {
    st.textContent = 'Could not request a batch: ' + err;
  } finally {
    setTimeout(() => { btn.disabled = false; }, 3000);
  }
});

function fillEpisodeDays(days) {
  const sel = $('#ep-day');
  const cur = sel.value;
  const html = '<option value="">all days</option>' +
    days.map(d => '<option value="' + esc(d) + '">' + esc(d) + '</option>').join('');
  if (sel.dataset.filled !== html) {
    sel.innerHTML = html;
    sel.dataset.filled = html;
    sel.value = cur;
  }
}

function episodeCard(e) {
  const visits = (e.visits || []).map(v => {
    const mins = v.duration_s ? Math.max(1, Math.round(v.duration_s / 60)) : 0;
    return '<a class="ep-visit" href="' + esc(v.image_url || '#') + '" '
      + 'target="_blank" rel="noopener">'
      + (v.image_url ? '<img src="' + esc(v.image_url) + '" alt="" loading="lazy">' : '')
      + '<span class="ep-visit-meta">' + esc(v.place || v.camera || '')
      + ' &middot; ' + esc(v.camera || '') + ' &middot; ' + esc(epClock(v.enter_time))
      + (mins ? ' &middot; ' + mins + ' min' : '') + '</span>'
      + (v.description_vlm
        ? '<span class="ep-visit-cap">' + esc(v.description_vlm) + '</span>' : '')
      + '</a>';
  }).join('');
  const tags = '<span class="tag">' + esc(e.anon_name || '') + '</span>'
    + (e.person_name ? '<span class="tag">' + esc(e.person_name) + '</span>' : '')
    + '<span class="tag">' + esc(e.day || '') + '</span>'
    + ((typeof e.link_confidence === 'number' && e.link_confidence < 0.99)
      ? '<span class="tag" title="confidence that this is one person across cameras">'
        + 'link ' + Math.round(e.link_confidence * 100) + '%</span>'
      : '');
  return '<div class="card ep-card">'
    + '<div class="ep-head">' + tags + '</div>'
    + '<div class="ep-narrative">' + esc(e.narrative || '') + '</div>'
    + (e.narrative_ar
      ? '<div class="ep-narrative ep-ar" dir="rtl" lang="ar">'
        + esc(e.narrative_ar) + '</div>'
      : '')
    + '<div class="ep-visits">' + visits + '</div>'
    + '</div>';
}

async function loadEpisodes() {
  const box = $('#ep-list');
  const st = $('#ep-status');
  const pager = $('#ep-pager');
  st.textContent = 'Loading…';
  const p = new URLSearchParams({
    limit: String(EP_LIMIT),
    offset: String((epPage - 1) * EP_LIMIT),
    with_visits: '1',
  });
  const cam = $('#ep-cam').value; if (cam) p.set('camera', cam);
  const day = $('#ep-day').value; if (day) p.set('day', day);
  try {
    const data = await api('/api/episodes?' + p.toString());
    const total = data.total || 0;
    epPages = Math.max(1, Math.ceil(total / EP_LIMIT));
    if (epPage > epPages) epPage = epPages;
    fillEpisodeDays(data.days || []);
    st.textContent = (total ? total + ' episode(s)' : '')
      + (data.note ? ' — ' + data.note : '');
    if (!data.items || !data.items.length) {
      box.innerHTML = '<div class="empty">No episodes yet. They appear once '
        + 'person captures are linked across cameras — check that '
        + '<b>CAMERA_PLACES</b> and <b>ADJACENCY</b> are filled in '
        + '<code>config/places.conf</code>.</div>';
      pager.classList.add('hidden');
      return;
    }
    pager.classList.remove('hidden');
    $('#ep-pageno').textContent = 'Page ' + epPage + ' of ' + epPages;
    box.innerHTML = data.items.map(episodeCard).join('');
  } catch (err) {
    st.textContent = 'Failed: ' + err;
    box.innerHTML = '';
    pager.classList.add('hidden');
  }
}

/* ------------- scenereader ADAPTIVE SCENES (L1) ------------------------------
   One card per SCENE: a camera's burst of activity, the objects that coexisted
   in it, and which of them MOVED (from each object's own Frigate trajectory).
   The narrative is composed DETERMINISTICALLY from those facts, in English and
   Arabic - no model. This is the level that keeps a moving dog/cow/truck in a
   crowded frame from being narrated as a person's journey.
   See plans/adaptive-scene-narrative.md. */
const AS_LIMIT = 20;
let asPage = 1;
let asPages = 1;

function reloadAdaptiveScenes() {
  asPage = 1;
  loadAdaptiveScenes();
}

function fillSceneDays(days) {
  const sel = $('#as-day');
  const cur = sel.value;
  const html = '<option value="">all days</option>' +
    days.map(d => '<option value="' + esc(d) + '">' + esc(d) + '</option>').join('');
  if (sel.dataset.filled !== html) {
    sel.innerHTML = html;
    sel.dataset.filled = html;
    sel.value = cur;
  }
}

function adaptiveSceneCard(s) {
  const objects = (s.objects || []).map(o =>
    '<span class="tag" title="own displacement ' + (Number(o.disp) || 0).toFixed(3) + '">'
    + esc(o.label || '') + (o.moved ? ' ● moved' : '') + '</span>').join('');
  const frames = (s.events || []).map(v =>
    '<a class="ep-visit" href="' + esc(v.image_url || '#') + '" target="_blank" rel="noopener">'
    + (v.image_url ? '<img src="' + esc(v.image_url) + '" alt="" loading="lazy">' : '')
    + '<span class="ep-visit-meta">' + esc(v.label || '') + ' &middot; '
    + esc(v.camera || '') + ' &middot; ' + esc(epClock(v.start_time)) + '</span></a>').join('');
  const movers = (s.movers || []).length
    ? '<span class="tag">moving: ' + esc(s.movers.join(', ')) + '</span>'
    : '<span class="tag">no movement</span>';
  return '<div class="card ep-card">'
    + '<div class="ep-head">'
    + '<span class="tag">' + esc(s.camera || '') + '</span>'
    + (s.place && s.place !== s.camera ? '<span class="tag">' + esc(s.place) + '</span>' : '')
    + '<span class="tag">' + esc(s.day || '') + '</span>'
    + movers
    + '</div>'
    + '<div class="ep-narrative">' + esc(s.narrative || '') + '</div>'
    + (s.narrative_ar
      ? '<div class="ep-narrative ep-ar" dir="rtl" lang="ar">' + esc(s.narrative_ar) + '</div>'
      : '')
    + '<div class="ep-head">' + objects + '</div>'
    + '<div class="ep-visits">' + frames + '</div>'
    + '</div>';
}

async function loadAdaptiveScenes() {
  const box = $('#as-list');
  const st = $('#as-status');
  const pager = $('#as-pager');
  st.textContent = 'Loading…';
  const p = new URLSearchParams({
    limit: String(AS_LIMIT),
    offset: String((asPage - 1) * AS_LIMIT),
    with_events: '1',
  });
  const cam = $('#as-cam').value; if (cam) p.set('camera', cam);
  const day = $('#as-day').value; if (day) p.set('day', day);
  try {
    const data = await api('/api/scenereader/scenes?' + p.toString());
    const total = data.total || 0;
    asPages = Math.max(1, Math.ceil(total / AS_LIMIT));
    if (asPage > asPages) asPage = asPages;
    fillSceneDays(data.days || []);
    st.textContent = (total ? total + ' scene(s)' : '')
      + (data.note ? ' — ' + data.note : '');
    if (!data.items || !data.items.length) {
      box.innerHTML = '<div class="empty">No scenes yet. They are rebuilt from '
        + 'Frigate captures automatically; run <code>--rebuild-scenes</code> to '
        + 'derive them now.</div>';
      pager.classList.add('hidden');
      return;
    }
    pager.classList.remove('hidden');
    $('#as-pageno').textContent = 'Page ' + asPage + ' of ' + asPages;
    box.innerHTML = data.items.map(adaptiveSceneCard).join('');
  } catch (err) {
    st.textContent = 'Failed: ' + err;
    box.innerHTML = '';
    pager.classList.add('hidden');
  }
}

$('#as-refresh').addEventListener('click', reloadAdaptiveScenes);
$('#as-cam').addEventListener('change', reloadAdaptiveScenes);
$('#as-day').addEventListener('change', reloadAdaptiveScenes);
$('#as-prev').addEventListener('click', () => {
  if (asPage > 1) { asPage--; loadAdaptiveScenes(); }
});
$('#as-next').addEventListener('click', () => {
  if (asPage < asPages) { asPage++; loadAdaptiveScenes(); }
});

/* ------------- scenewatch scene descriptions (paged + time-filtered) ---------
   One row per captioned frame (scenewatch's two-frame motion gate; `reason`
   says whether motion or the periodic baseline triggered it). The tab lists the
   description + camera + time; the stored frame is the card image and opens in
   a lightbox. Rows captured while STORE_IMAGES=false carry no image. */
const SC_PAGE = 24;
let scPage = 1;
let scPages = 1;
let scMeta = {};              // scene id -> full record (for the lightbox)

function reloadScenes() {
  scPage = 1;
  scMeta = {};
  loadScenes();
}
$('#sc-refresh').addEventListener('click', reloadScenes);
$('#sc-cam').addEventListener('change', reloadScenes);
$('#sc-reason').addEventListener('change', reloadScenes);
$('#sc-tier').addEventListener('change', reloadScenes);
$('#sc-sort').addEventListener('change', reloadScenes);
wireTimeControls('sc', reloadScenes);
$('#sc-prev').addEventListener('click', () => {
  if (scPage > 1) { scPage--; loadScenes(); }
});
$('#sc-next').addEventListener('click', () => {
  if (scPage < scPages) { scPage++; loadScenes(); }
});

async function loadScenes() {
  const box = $('#sc-list');
  const st = $('#sc-status');
  st.textContent = 'Loading…';
  hidePager('sc');
  try {
    const { after, before } = timeRange($('#sc-time'), $('#sc-from'), $('#sc-to'));
    const p = new URLSearchParams({
      limit: String(SC_PAGE),
      offset: String((scPage - 1) * SC_PAGE),
    });
    const cam = $('#sc-cam').value; if (cam) p.set('camera', cam);
    const rsn = $('#sc-reason').value; if (rsn) p.set('reason', rsn);
    // "Show" = how important a scene must be. Default 'high' keeps the tab on
    // the handful of rows worth seeing; 'all' is the "dig deeper" escape.
    const tier = $('#sc-tier').value;
    if (tier && tier !== 'all') p.set('min_tier', tier);
    p.set('sort', $('#sc-sort').value || 'importance');
    if (after) p.set('after', String(after));
    if (before) p.set('before', String(before));
    const data = await api('/api/scenes?' + p.toString());
    const total = data.total || 0;
    scPages = Math.max(1, Math.ceil(total / SC_PAGE));
    if (scPage > scPages) scPage = scPages;
    // `note` explains an empty list (DB missing, or scenewatch never captioned).
    st.textContent = (total ? total + ' description(s)' : '') +
      (data.note ? (total ? ' — ' : '') + data.note : '');
    box.innerHTML = '';
    if (!data.items.length) {
      // Spell out the way out when the importance filter is the reason the
      // list is empty - otherwise a quiet window looks like a broken tab.
      const hint = ($('#sc-tier').value === 'high')
        ? '<br>No <b>important</b> scenes in this range. Set <b>Show</b> to '
          + '&ldquo;important + normal&rdquo; or &ldquo;everything&rdquo; '
          + 'to dig deeper.'
        : '';
      box.innerHTML = '<div class="empty">No scene descriptions.' + hint + '</div>';
      hidePager('sc');
      return;
    }
    const frag = document.createElement('div');
    frag.innerHTML = data.items.map(sceneCard).join('');
    $$('.card', frag).forEach(c => box.appendChild(c));
    renderPager('sc', scPage, scPages);
  } catch (e) {
    st.textContent = '';
    box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
    hidePager('sc');
  }
}

/* Importance (writer-side, see score_importance in scenewatch.py): `tier` is
   high/normal/low and `importance` the 0-100 score behind it. */
function sceneTags(s) {
  const tier = s.tier || 'normal';
  const label = tier === 'high' ? 'important' : tier;
  return '<span class="tag tier-' + esc(tier) + '" title="Importance score ' +
      Number(s.importance || 0) + '/100 (scored when the caption was written)">' +
      esc(label) + ' ' + Number(s.importance || 0) + '</span>' +
    '<span class="tag ' + (s.reason === 'baseline' ? 'baseline' : 'motion') + '">' +
      esc(s.reason || '') + '</span>';
}

function sceneCard(s) {
  scMeta[s.id] = s;
  const img = s.has_image
    ? '<div class="sc-imgwrap"><img class="sc-img" data-id="' + s.id + '" ' +
      'src="' + esc(s.image_url) + '" alt="scene frame" loading="lazy"></div>'
    : '<div class="sc-imgwrap noimg">no image stored</div>';
  return '<div class="card">' +
    '<div class="thumb">' + img + '</div>' +
    '<div class="scene-desc">' + esc(s.description || '') + '</div>' +
    '<div class="meta">' +
      sceneTags(s) +
      '<span class="tag">' + esc(s.camera) + '</span>' +
      '<span class="muted">' + esc(fmtDT(s.captured_at)) + '</span>' +
      (s.motion_frac ? '<span class="muted">motion ' +
        (Number(s.motion_frac) * 100).toFixed(2) + '%</span>' : '') +
      (s.latency_ms ? '<span class="muted">' + Number(s.latency_ms) + ' ms</span>' : '') +
    '</div>' +
  '</div>';
}

/* -------- scene lightbox: click a card image to view the frame larger ------ */
function openSceneLightbox(id) {
  const s = scMeta[id];
  if (!s || !s.has_image) return;
  const img = $('#sc-lb-img');
  $('#sc-lb-title').textContent = 'Scene #' + id;
  $('#sc-lb-meta').innerHTML =
    sceneTags(s) +
    '<span class="tag">' + esc(s.camera || '') + '</span>' +
    '<span class="muted">' + esc(fmtDT(s.captured_at)) + '</span>' +
    '<div class="lb-desc">' + esc(s.description || '') + '</div>';
  img.onerror = () => { img.onerror = null; };
  $('#sc-lightbox').classList.remove('hidden');
  document.body.classList.add('lb-open');
  img.src = s.image_url;
}

function closeSceneLightbox() {
  const lb = $('#sc-lightbox');
  if (!lb || lb.classList.contains('hidden')) return;
  lb.classList.add('hidden');
  document.body.classList.remove('lb-open');
  const img = $('#sc-lb-img');
  img.onerror = null;
  img.removeAttribute('src');
}

// Delegated so it survives every re-render (cards are replaced on each load).
$('#sc-list').addEventListener('click', (e) => {
  const el = e.target.closest('.sc-img');
  if (el && el.dataset.id) openSceneLightbox(el.dataset.id);
});
$('#sc-lb-close').addEventListener('click', closeSceneLightbox);
$('#sc-lightbox').addEventListener('click', (e) => {
  if (e.target.closest('[data-sc-close]')) closeSceneLightbox();   // backdrop tap
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeSceneLightbox();
});

/* ---------------- bandwidth usage (self + embedded admin view) ----------------
   The portal counts the REAL bytes it sends each logged-in user (Live video,
   event clips, snapshots/JSON) as daily counters - see portal/usage.py. Every
   user sees their OWN totals in the footer chip and the Account tab's usage card
   (from /api/usage); an admin also sees everyone's (from /api/admin/usage).
   NOTE: the separate Usage TAB was removed - usage now lives INSIDE the Account
   (User) tab, reachable by clicking the username. */
const USAGE_KINDS = [['live', 'Live'], ['events', 'Events video'],
                     ['other', 'Other']];

function fmtBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return n + ' B';
  const units = ['KB', 'MB', 'GB', 'TB', 'PB'];
  let i = -1;
  do { n /= 1024; i++; } while (n >= 1024 && i < units.length - 1);
  return (n >= 100 ? n.toFixed(0) : n.toFixed(1)) + ' ' + units[i];
}

/* ---- per-user quota (used / quota) progress bar ----
   state.me.quota_bytes (from /api/me) is the account's data budget; the USED
   total is state.usage.total.bytes (from /api/usage). The bar lives in the
   large-screen footer (#quota-footer) AND in the phone collapsed menu
   (#quota-nav) - CSS shows only one at a time. 0 = unlimited (bar hidden). */
const GB_BYTES = 1024 * 1024 * 1024;

function quotaInfo() {
  const quota = Number((state.me && state.me.quota_bytes) || 0);
  const used = Number((state.usage && state.usage.total &&
                       state.usage.total.bytes) || 0);
  const pct = quota > 0 ? (used / quota) * 100 : 0;
  return { used: used, quota: quota, pct: pct };
}

function quotaText(info) {
  const p = info.pct >= 10 ? info.pct.toFixed(0) : info.pct.toFixed(1);
  return fmtBytes(info.used) + ' / ' + fmtBytes(info.quota) + ' (' + p + '%)';
}

function renderQuotaBars() {
  const info = quotaInfo();
  ['quota-footer', 'quota-nav'].forEach((id) => {
    const box = $('#' + id);
    if (!box) return;
    if (!info.quota) { box.classList.add('hidden'); return; }
    box.classList.remove('hidden');
    const txt = $('#' + id + '-text');
    if (txt) txt.textContent = quotaText(info);
    const fill = $('#' + id + '-fill');
    if (fill) {
      fill.style.width = Math.min(100, info.pct).toFixed(1) + '%';
      fill.classList.toggle('warn', info.pct >= 80 && info.pct < 100);
      fill.classList.toggle('over', info.pct >= 100);
    }
  });
}

/* Self-view table: today / 7d / 30d / all time, by kind plus a total column. */
function usageSelfTable(b) {
  let head = '<tr><th></th>';
  USAGE_KINDS.forEach(([, name]) => { head += '<th>' + esc(name) + '</th>'; });
  head += '<th>Total</th></tr>';
  const row = (label, x) => {
    let cells = '';
    USAGE_KINDS.forEach(([k]) => { cells += '<td>' + fmtBytes(x[k]) + '</td>'; });
    return '<tr><th>' + esc(label) + '</th>' + cells +
      '<td>' + fmtBytes(x.bytes) + '</td></tr>';
  };
  return '<div class="usage-scroll"><table class="usage-table">' +
    '<thead>' + head + '</thead><tbody>' +
    row('Today', b.today || {}) +
    row('Last 7 days', b.d7 || {}) +
    row('Last 30 days', b.d30 || {}) +
    row('All time', b.total || {}) +
    '</tbody></table></div>';
}

function usageNote(retentionDays) {
  return '<p class="usage-note">Bytes the portal sent to this account - Live ' +
    'video, event clips, snapshots/JSON - counted daily. History kept ' +
    esc(retentionDays || 180) + ' days.</p>';
}

function renderUsageSelf() {
  if (!state.usage) return;
  renderQuotaBars();
}

async function loadUsageSelf() {
  try { state.usage = await api('/api/usage'); } catch (e) { return; }
  renderUsageSelf();
}

/* Admin: every user's usage, rendered into the Account tab's admin card. */
async function loadAllUsage() {
  const box = $('#mu-all-usage');
  if (!box) return;
  let data;
  try { data = await api('/api/admin/usage'); }
  catch (e) { box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>'; return; }
  const users = (data && data.users) || [];
  const r = (data && data.retention_days) || 180;
  if (!users.length) {
    box.innerHTML = '<div class="empty">No usage recorded yet.</div>';
    return;
  }
  const head = '<tr><th>User</th><th>Live (today)</th><th>Events (today)</th>' +
    '<th>Other (today)</th><th>Today</th><th>30 days</th><th>All time</th></tr>';
  const rows = users.map(u => {
    const t = u.today || {}, d30 = u.d30 || {}, tot = u.total || {};
    return '<tr><td>' + esc(u.username) + '</td>' +
      '<td>' + fmtBytes(t.live) + '</td>' +
      '<td>' + fmtBytes(t.events) + '</td>' +
      '<td>' + fmtBytes(t.other) + '</td>' +
      '<td>' + fmtBytes(t.bytes) + '</td>' +
      '<td>' + fmtBytes(d30.bytes) + '</td>' +
      '<td>' + fmtBytes(tot.bytes) + '</td></tr>';
  }).join('');
  box.innerHTML = '<div class="usage-scroll"><table class="usage-table">' +
    '<thead>' + head + '</thead><tbody>' + rows + '</tbody></table></div>' +
    usageNote(r);
}

/* ---------------- app access: `tab_*` permissions ----------------
   Access is expressed as `tab_<name>` permission keys (e.g. `tab_live`), the
   catalogue in TABS below. An ADMIN implicitly has EVERY tab - current AND
   future - so admins never depend on the stored keys. A non-admin gets exactly
   the keys an admin checked for them (EMPTY = no tabs). Accounts live in the
   portal's own users DB (portal/userstore.py), so changes apply at once with NO
   container restart. Keep TABS in sync with AVAILABLE_TABS in userstore.py. */
const TABS = [['live', 'Live'], ['events', 'Events'], ['fire', 'Fire alerts'],
              ['episodes', 'Episodes'], ['adaptive', 'Scenes'],
              ['scenes', 'Scene log'], ['notifications', 'Notifications']];
const USER_VIEWS = TABS.map(t => t[0]);
function tabKey(v) { return 'tab_' + v; }

function isAdmin() { return !!(state.me && state.me.is_admin); }

/* A view is reachable when its `tab_*` key is granted. The Account tab ('user')
   is always available; Debug/Manage are admin-only. */
function viewAllowed(v) {
  if (isAdmin()) return true;
  if (v === 'debug' || v === 'manage') return false;
  if (USER_VIEWS.indexOf(v) < 0) return true;      // 'user'
  const p = (state.me && state.me.permissions) || [];
  return p.indexOf(tabKey(v)) >= 0;
}

/* Show only the granted tabs, and the admin-only Manage/Debug links (the server
   is still the real gate for every endpoint). */
function applyNavPermissions() {
  const admin = isAdmin();
  const dbg = $('#nav-debug');
  if (dbg) dbg.classList.toggle('hidden', !admin);
  const man = $('#nav-manage');
  if (man) man.classList.toggle('hidden', !admin);
  const p = (state.me && state.me.permissions) || [];
  $$('#nav a[data-view]').forEach(a => {
    const v = a.dataset.view;
    if (USER_VIEWS.indexOf(v) < 0) return;   // account/admin links untouched
    a.classList.toggle('hidden', !admin && p.indexOf(tabKey(v)) < 0);
  });
}

/* Generated avatar so an account without a photo still shows something. */
function avatarPlaceholder(name) {
  const ch = (String(name || '?').trim().charAt(0) || '?').toUpperCase();
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160">' +
    '<rect width="160" height="160" fill="#1b222c"/>' +
    '<text x="80" y="104" font-size="76" text-anchor="middle" fill="#7d8b99" ' +
    'font-family="sans-serif">' + ch + '</text></svg>';
  return 'data:image/svg+xml,' + encodeURIComponent(svg);
}

/* Read an image file and downscale it (<=256 px, JPEG) so the stored data URL
   stays small. Falls back to the raw data URL if the canvas path fails. */
function fileToDataUrl(file, maxPx) {
  maxPx = maxPx || 256;
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('read failed'));
    reader.onload = () => {
      const raw = String(reader.result || '');
      const img = new Image();
      img.onerror = () => reject(new Error('not a valid image'));
      img.onload = () => {
        try {
          const s = Math.min(1, maxPx / Math.max(img.width, img.height));
          const w = Math.max(1, Math.round(img.width * s));
          const h = Math.max(1, Math.round(img.height * s));
          const c = document.createElement('canvas');
          c.width = w; c.height = h;
          c.getContext('2d').drawImage(img, 0, 0, w, h);
          resolve(c.toDataURL('image/jpeg', 0.85));
        } catch (e) { resolve(raw); }
      };
      img.src = raw;
    };
    reader.readAsDataURL(file);
  });
}

function renderUserPhoto(name) {
  const img = $('#user-photo-img');
  if (img) img.src = state.userPhoto || avatarPlaceholder(name);
  const clear = $('#user-photo-clear');
  if (clear) clear.disabled = !state.userPhoto;
}

/* Render the `tab_*` permission checkboxes into `#<sel>`.

   `adminAll` (the selected user is an administrator) shows every box checked
   and disabled - an admin always has every tab, current and future. Otherwise
   a box is checked exactly when its `tab_*` key is stored (an EMPTY permission
   list = nothing checked). */
function renderTabBoxes(sel, perms, editable, adminAll) {
  const box = $('#' + sel);
  if (!box) return;
  const list = perms || [];
  box.innerHTML = TABS.map(([v, label]) => {
    const key = tabKey(v);
    const checked = adminAll || list.indexOf(key) >= 0;
    return '<label class="check" title="' + esc(key) + '">' +
      '<input type="checkbox" value="' + esc(key) + '"' +
      (checked ? ' checked' : '') +
      ((editable && !adminAll) ? '' : ' disabled') + '> ' +
      esc(label) + '</label>';
  }).join('');
}

/* The Account tab shows the signed-in user's OWN profile. */
function renderUserProfile(p) {
  state.userPhoto = p.photo || '';
  const title = $('#user-title');
  if (title) title.textContent = '#' + (p.username || '');
  $('#user-username').value = p.username || '';
  $('#user-display').value = p.display_name || '';
  $('#user-camera').value = p.default_camera || '';
  renderUserPhoto(p.username);
  const save = $('#user-save');
  if (save) save.disabled = false;
}

/* Save your OWN profile (display name / default camera / photo). Permissions,
   role and active flag are admin-managed on the Manage tab. */
async function saveUserProfile() {
  const target = state.me && state.me.username;
  if (!target) return;
  const btn = $('#user-save');
  btn.disabled = true;
  try {
    const p = await api('/api/users/' + encodeURIComponent(target),
                        { method: 'PATCH', body: {
      display_name: $('#user-display').value.trim(),
      default_camera: $('#user-camera').value.trim(),
      photo: state.userPhoto || '',
    }});
    renderUserProfile(p);
    toast('Saved');
  } catch (e) {
    toast('Could not save: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

async function saveUserPassword() {
  const msg = $('#pw-msg');
  const target = state.me && state.me.username;
  if (!target) return;
  const cur = $('#pw-current').value;
  const nw = $('#pw-new').value;
  const cf = $('#pw-confirm').value;
  msg.textContent = '';
  if (!nw || nw.length < 6) {
    msg.textContent = 'New password must be at least 6 characters.'; return;
  }
  if (nw !== cf) {
    msg.textContent = 'New password and confirmation do not match.'; return;
  }
  // Changing YOUR OWN password always requires the current one.
  const body = { new_password: nw, current_password: cur };
  const btn = $('#pw-save');
  btn.disabled = true;
  try {
    await api('/api/users/' + encodeURIComponent(target) + '/password',
              { method: 'POST', body });
    $('#pw-current').value = ''; $('#pw-new').value = ''; $('#pw-confirm').value = '';
    msg.textContent = 'Password updated.';
  } catch (e) {
    msg.textContent = 'Could not update: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* The Account tab's usage card - always the CALLER's own totals. */
async function loadUserUsage() {
  const box = $('#user-usage');
  if (!box) return;
  box.innerHTML = '<div class="empty">Loading\u2026</div>';
  try {
    const data = await api('/api/usage');
    state.usage = data;
    renderUsageSelf();
    const info = quotaInfo();
    const qline = info.quota
      ? '<p class="quota-summary">Quota: <b>' + esc(quotaText(info)) +
        '</b></p>' : '';
    box.innerHTML = qline + usageSelfTable(data) +
      usageNote(data.retention_days);
  } catch (e) {
    box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
  }
}

/* ---- Manage tab (admin): user administration + all-users usage ----
   The FIRST card creates accounts (username + password) and, when a user is
   SELECTED, edits their password and checks the `tab_*` permissions they get.
   The second card shows every user's bandwidth usage. Applied immediately -
   no container restart. */

function userRow(u) {
  const photo = u.photo
    ? '<img class="user-thumb" src="' + esc(u.photo) + '" alt="">'
    : '<span class="user-thumb empty-av">' +
      esc((String(u.username || '?').charAt(0) || '?').toUpperCase()) + '</span>';
  const perms = u.is_admin ? 'all (admin)'
    : ((u.permissions && u.permissions.length)
       ? u.permissions.join(', ') : 'none');
  const hash = u.password_hash || '';
  const hashShort = hash.length > 22 ? hash.slice(0, 22) + '\u2026' : hash;
  const selected = (u.username || '').toLowerCase() ===
    (state.manageTarget || '').toLowerCase();
  return '<tr' + (selected ? ' class="active"' : '') + '>' +
    '<td><button class="user-link" type="button" data-mu-open="' +
      esc(u.username) + '">' + esc(u.username) + '</button></td>' +
    '<td>' + esc(u.display_name || '') + '</td>' +
    '<td class="user-cell-photo">' + photo + '</td>' +
    '<td>' + (u.is_admin ? 'yes' : '') + '</td>' +
    '<td>' + (u.is_active === false ? 'disabled' : 'yes') + '</td>' +
    '<td>' + esc(perms) + '</td>' +
    '<td>' + (u.quota_bytes ? fmtBytes(u.quota_bytes) : '\u2014') + '</td>' +
    '<td class="user-hash" title="' + esc(hash) + '">' + esc(hashShort) + '</td>' +
    '</tr>';
}

async function loadManageUsers() {
  const box = $('#mu-list');
  if (!box) return;
  let data;
  try { data = await api('/api/users'); }
  catch (e) { box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>'; return; }
  state.users = data.users || [];
  if (!state.users.length) {
    box.innerHTML = '<div class="empty">No users.</div>';
    return;
  }
  box.innerHTML = '<div class="usage-scroll"><table class="usage-table user-table">' +
    '<thead><tr><th>Username</th><th>Display name</th><th>Photo</th>' +
    '<th>Admin</th><th>Active</th><th>Permissions</th><th>Quota</th>' +
    '<th>Password hash</th></tr></thead>' +
    '<tbody>' + state.users.map(userRow).join('') + '</tbody></table></div>';
}

/* Render the edit panel for the SELECTED user (password + tab_* permissions). */
function renderManageEdit() {
  const panel = $('#mu-edit');
  if (!panel) return;
  const u = (state.users || []).find(x =>
    (x.username || '').toLowerCase() ===
    (state.manageTarget || '').toLowerCase());
  if (!u) { panel.classList.add('hidden'); return; }
  panel.classList.remove('hidden');
  const meName = (state.me && state.me.username || '').toLowerCase();
  const isSelf = u.username.toLowerCase() === meName;
  $('#mu-edit-name').textContent = u.username;
  $('#mu-pw').value = '';
  // Changing your OWN password must confirm the CURRENT one (server rule).
  $('#mu-pw-current').value = '';
  $('#mu-pw-current-wrap').classList.toggle('hidden', !isSelf);
  $('#mu-admin').checked = !!u.is_admin;
  $('#mu-active').checked = u.is_active !== false;
  // You cannot deactivate or delete your OWN account (the server blocks it too).
  $('#mu-active').disabled = isSelf || !!u.is_admin;
  $('#mu-delete').disabled = isSelf;
  // Your OWN administrator flag is NOT editable from the portal (too easy to
  // lock yourself out); the server also rejects a self-demotion.
  $('#mu-admin').disabled = isSelf;
  const gb = Number(u.quota_bytes || 0) / GB_BYTES;
  $('#mu-quota').value = String(Math.round(gb * 100) / 100);
  renderTabBoxes('mu-perms', u.permissions, true, !!u.is_admin);
  $('#mu-edit-msg').textContent = '';
}

function selectManageUser(username) {
  state.manageTarget = username || null;
  loadManageUsers().then(renderManageEdit);
}

async function loadManage() {
  if (!isAdmin()) return;
  await loadManageUsers();
  if (!state.manageTarget ||
      !state.users.some(u => u.username.toLowerCase() ===
        state.manageTarget.toLowerCase())) {
    state.manageTarget = state.users.length ? state.users[0].username : null;
  }
  renderManageEdit();
  loadAllUsage();
}

/* ---- Manage tab wiring ---- */
$('#mu-add').addEventListener('click', async () => {
  const msg = $('#mu-msg');
  const btn = $('#mu-add');
  msg.textContent = '';
  const username = $('#mu-new-username').value.trim();
  const password = $('#mu-new-password').value;
  if (!username || !password) {
    msg.textContent = 'Username and password are required.'; return;
  }
  btn.disabled = true;
  try {
    // The backend hashes the password (pbkdf2). The new account starts with NO
    // tab access - check the tabs for it below.
    const created = await api('/api/users',
                              { method: 'POST', body: { username, password } });
    $('#mu-new-username').value = ''; $('#mu-new-password').value = '';
    toast('User created');
    state.manageTarget = (created && created.username) || username;
    await loadManageUsers();
    renderManageEdit();
  } catch (e) {
    msg.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
});
$('#mu-list').addEventListener('click', (e) => {
  const b = e.target.closest('[data-mu-open]');
  if (b) selectManageUser(b.dataset.muOpen);
});
$('#mu-pw-save').addEventListener('click', async () => {
  const msg = $('#mu-edit-msg');
  const pw = $('#mu-pw').value;
  msg.textContent = '';
  if (!pw || pw.length < 6) {
    msg.textContent = 'New password must be at least 6 characters.'; return;
  }
  const body = { new_password: pw };
  // Your OWN account needs the current password; resetting ANOTHER user's does
  // not (the backend hashes it either way).
  if (!$('#mu-pw-current-wrap').classList.contains('hidden')) {
    body.current_password = $('#mu-pw-current').value;
  }
  const btn = $('#mu-pw-save');
  btn.disabled = true;
  try {
    await api('/api/users/' + encodeURIComponent(state.manageTarget) + '/password',
              { method: 'POST', body });
    $('#mu-pw').value = '';
    $('#mu-pw-current').value = '';
    msg.textContent = 'Password updated.';
  } catch (e) {
    msg.textContent = 'Could not update: ' + e.message;
  } finally {
    btn.disabled = false;
  }
});
$('#mu-save').addEventListener('click', async () => {
  if (!state.manageTarget) return;
  const msg = $('#mu-edit-msg');
  msg.textContent = '';
  const body = {
    permissions: $$('#mu-perms input:checked').map(i => i.value),
  };
  // Skipped when the control is disabled (your OWN admin flag / active state).
  if (!$('#mu-admin').disabled) body.is_admin = $('#mu-admin').checked;
  if (!$('#mu-active').disabled) body.is_active = $('#mu-active').checked;
  const qgb = parseFloat($('#mu-quota').value);
  if (!isNaN(qgb) && qgb >= 0) body.quota_bytes = Math.round(qgb * GB_BYTES);
  const btn = $('#mu-save');
  btn.disabled = true;
  try {
    await api('/api/users/' + encodeURIComponent(state.manageTarget),
              { method: 'PATCH', body });
    toast('Saved');
    await loadManageUsers();
    renderManageEdit();
  } catch (e) {
    msg.textContent = 'Could not save: ' + e.message;
  } finally {
    btn.disabled = false;
  }
});
$('#mu-delete').addEventListener('click', async () => {
  const msg = $('#mu-edit-msg');
  const name = state.manageTarget;
  if (!name) return;
  if (!window.confirm('Delete user "' + name + '"? This cannot be undone.')) return;
  const btn = $('#mu-delete');
  btn.disabled = true;
  try {
    await api('/api/users/' + encodeURIComponent(name), { method: 'DELETE' });
    toast('Deleted ' + name);
    state.manageTarget = null;
    await loadManage();
  } catch (e) {
    msg.textContent = 'Could not delete: ' + e.message;
    btn.disabled = false;
  }
});
$('#mu-usage-refresh').addEventListener('click', loadAllUsage);

/* ---------------- Account (User) tab ----------------
   Always the SIGNED-IN user's own profile (display name / photo / password) with
   the embedded usage card. Admin management of OTHER accounts is the Manage tab. */
async function loadUserView() {
  if (!state.me) return;
  state.userTarget = state.me.username;
  let profile;
  try { profile = await api('/api/users/' + encodeURIComponent(state.me.username)); }
  catch (e) { toast('Could not load account: ' + e.message); return; }
  renderUserProfile(profile);
  loadUserUsage();
}

/* ---- Account tab wiring ---- */
$('#user-save').addEventListener('click', saveUserProfile);
$('#pw-save').addEventListener('click', saveUserPassword);
$('#user-usage-refresh').addEventListener('click', loadUserUsage);
$('#user-photo-pick').addEventListener('click', () => $('#user-photo-file').click());
$('#user-photo-file').addEventListener('change', async (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) {
    try {
      state.userPhoto = await fileToDataUrl(f);
      renderUserPhoto($('#user-username').value);
    } catch (err) { toast('Could not read that image'); }
  }
  e.target.value = '';
});
$('#user-photo-clear').addEventListener('click', () => {
  state.userPhoto = '';
  renderUserPhoto($('#user-username').value);
});
// The username (topbar chip / collapsed menu) and the footer usage chip all
// open the Account tab for YOURSELF.
function gotoSelfTab() {
  setNavOpen(false);
  if ((location.hash || '') === '#/user') loadUserView();
  else location.hash = '#/user';
}
$('#user-chip').addEventListener('click', gotoSelfTab);
$('#nav-user').addEventListener('click', gotoSelfTab);
$('#quota-footer').addEventListener('click', gotoSelfTab);
$('#quota-nav').addEventListener('click', gotoSelfTab);

/* ---------------- admin Debug: logs of containers (selectable) ---------------- */
/* The read-only `logs` sidecar stays idle; the portal pulls its container
   list + log tails ONLY when this tab is opened (and on Refresh / tail or
   container change). The Container dropdown lets the admin view ONE container's
   logs instead of every container at once ("All containers" is the default).
   Every card ALSO carries its own refresh button so one container's tail can be
   reloaded without re-fetching the rest; lines render newest-first. */
/* isAdmin() is defined once, with the Account (User) tab above, and shared by
   this Debug section. */

/* Fill the Container dropdown from /api/admin/containers (no log tails pulled
   just to build the list). Keeps the previously selected container when it
   still exists; otherwise falls back to "All containers". */
async function refreshContainerSelect() {
  const sel = $('#dbg-container');
  const prev = sel ? sel.value : '';
  const data = await api('/api/admin/containers');
  const list = (data && data.containers) || [];
  sel.innerHTML = '<option value="">All containers</option>' +
    list.map(c => '<option value="' + esc(c.name) + '">' + esc(c.name) + '</option>').join('');
  sel.value = (prev && list.some(c => c.name === prev)) ? prev : '';
}

async function loadDebug() {
  const box = $('#dbg-list');
  const st = $('#dbg-status');
  if (!isAdmin()) {
    box.innerHTML = '<div class="empty">Admin only</div>';
    return;
  }
  st.textContent = 'Loading container logs\u2026';
  box.innerHTML = '';
  try {
    await refreshContainerSelect();        // fresh container list for the dropdown
    const tailSel = $('#dbg-tail');
    const tail = parseInt(tailSel && tailSel.value, 10) || 200;
    const sel = $('#dbg-container');
    const name = sel ? (sel.value || '') : '';
    let url = '/api/admin/logs?tail=' + tail;
    if (name) url += '&name=' + encodeURIComponent(name);
    const data = await api(url);
    renderDebug(data);
    const n = (data.containers ? data.containers.length : 0);
    st.textContent = name
      ? 'container \u2018' + name + '\u2019 \u00b7 last ' + data.tail + ' lines'
      : n + ' container(s) \u00b7 last ' + data.tail + ' lines each';
  } catch (e) {
    st.textContent = '';
    box.innerHTML = '<div class="empty">' + esc(e.message) + '</div>';
  }
}

function renderDebug(data) {
  const box = $('#dbg-list');
  const frag = document.createElement('div');
  frag.innerHTML = (data.containers || []).map(dbgCard).join('');
  box.innerHTML = '';
  Array.from(frag.children).forEach(c => box.appendChild(c));
}

/* Docker logs come oldest-first; the UI shows the NEWEST line at the top (the
   reader almost always wants the latest output). Reverse the tail lines. */
function newestFirst(text) {
  const lines = String(text == null ? '' : text).split('\n');
  if (lines.length && lines[lines.length - 1] === '') lines.pop();  // trailing \n
  lines.reverse();
  return lines.join('\n');
}

function dbgCard(c) {
  const running = c.state === 'running';
  const name = c.name || '';
  const dot = '<span class="dbg-dot' + (running ? ' on' : '')
    + '" title="' + esc(c.state || 'stopped') + '"></span>';
  return '<div class="dbg-card" data-dbg-card="' + esc(name) + '">' +
    '<div class="dbg-head">' + dot +
      '<span class="dbg-name">' + esc(name || '?') + '</span>' +
      '<span class="dbg-meta">' + esc(c.state || '') + '</span>' +
      '<span class="dbg-meta">' + esc(c.status || '') + '</span>' +
      '<span class="dbg-img">' + esc(c.image || '') + '</span>' +
      '<button class="dbg-refresh" type="button" data-dbg-refresh="' + esc(name) +
        '" title="Reload only this container\u0027s logs" ' +
        'aria-label="Reload ' + esc(name) + ' logs">&#8635;</button>' +
    '</div>' +
    (c.error
      ? '<div class="dbg-err">log unavailable: ' + esc(c.error) + '</div>'
      : '<pre class="dbg-logs">' + esc(newestFirst(c.logs || '')) + '</pre>') +
  '</div>';
}

/* Refresh ONE container's log tail and swap ONLY that card - the admin is never
   forced to re-fetch every container (the global Refresh still does all). */
async function refreshOneContainerLog(name, btn) {
  if (!name) return;
  const card = btn ? btn.closest('.dbg-card') : null;
  const tailSel = $('#dbg-tail');
  const tail = parseInt(tailSel && tailSel.value, 10) || 200;
  if (btn) btn.disabled = true;
  try {
    const data = await api('/api/admin/logs?tail=' + tail +
                           '&name=' + encodeURIComponent(name));
    const c = (data.containers || [])[0];
    if (c && card) {
      const tmp = document.createElement('div');
      tmp.innerHTML = dbgCard(c);
      if (tmp.firstElementChild) card.replaceWith(tmp.firstElementChild);
    }
  } catch (e) {
    toast('Log refresh failed: ' + (e && e.message ? e.message : e));
    if (btn && document.body.contains(btn)) btn.disabled = false;
  }
}

$('#dbg-refresh').addEventListener('click', loadDebug);
$('#dbg-tail').addEventListener('change', loadDebug);
$('#dbg-container').addEventListener('change', loadDebug);
// Per-container refresh (delegated: cards are re-rendered on every load).
$('#dbg-list').addEventListener('click', (e) => {
  const b = e.target.closest('.dbg-refresh');
  if (b) refreshOneContainerLog(b.dataset.dbgRefresh, b);
});

/* ---------------- start ---------------- */
boot();
