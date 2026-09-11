/* mazr3a CCTV portal - NOTIFICATION-ONLY service worker.
 *
 * WHY THIS EXISTS: Android Chrome cannot construct `new Notification()` (it
 * throws `Illegal constructor`), so browser notifications on Android MUST be
 * raised through ServiceWorkerRegistration.showNotification(). This worker
 * therefore exists ONLY to host those notifications and handle their click.
 *
 * WHAT IT DELIBERATELY DOES NOT DO:
 *   - NO `fetch` handler and NO Cache Storage use of any kind. It cannot cache,
 *     serve or shadow any asset, so it has no interaction with the portal's
 *     content-hash cache-busting policy (see .roo/rules/portal-cache-busting.md)
 *     and can never serve a stale app.js/style.css.
 *   - NO `push` handler: the portal does not use Web Push; notifications are
 *     derived from the portal's own feed by polling in the page (app.js).
 *
 * Served at the STABLE, unfingerprinted URL /sw.js by portal/app.py (a service
 * worker registration needs a stable URL; a hashed URL would register a new
 * worker on every release and break the update flow).
 */
'use strict';

self.addEventListener('install', () => {
  // Take over immediately - there is no cached state to migrate.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

/* Tapping a notification focuses an existing portal tab (navigating it to the
   notification's deep link) or opens a new one. `data.url` is an SPA hash link
   like "#/fire", resolved against the worker scope. */
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const raw = (event.notification.data && event.notification.data.url) || '/';
  let target = '/';
  try { target = new URL(raw, self.registration.scope).href; }
  catch (e) { target = self.registration.scope; }
  event.waitUntil((async () => {
    const windows = await self.clients.matchAll(
      { type: 'window', includeUncontrolled: true });
    for (const client of windows) {
      if (client.url && 'focus' in client) {
        try { await client.focus(); } catch (e) { /* ignore */ }
        if ('navigate' in client) {
          try { await client.navigate(target); } catch (e) { /* ignore */ }
        }
        return;
      }
    }
    if (self.clients.openWindow) return self.clients.openWindow(target);
  })());
});
