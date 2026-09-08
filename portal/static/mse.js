/* go2rtc MSE over WebSocket player (mirrors Frigate 0.17's MsePlayer).

Frigate proxies go2rtc's MSE websocket at  /live/mse/api/ws?src=<cam>.
The server first sends a JSON message, e.g. {"type":"mse","value":"..."} where
value is the codecs/MIME (or {"type":"error",...}); afterwards it streams
binary fMP4 frames that are appended to a SourceBuffer. H.264 remux only - the
browser does the decode. One connection at a time; stop() closes the socket so
go2rtc releases the camera pull. Fallback to snapshot is handled by the caller.
*/
'use strict';

class G2MSEPlayer {
  constructor(video) {
    this.video = video;
    this.connected = false;
    this.ws = null;
    this.ms = null;
    this.sb = null;
    this.queue = [];
    this._evict = null;
    this.onstatus = null;   // (msg) => void
    this.onerror = null;    // (err) => void - called only on real failure
  }

  get supported() {
    const MS = window.ManagedMediaSource || window.MediaSource;
    if (!MS) return false;
    try { return MS.isTypeSupported('video/mp4'); } catch (e) { return false; }
  }

  _status(msg) { if (this.onstatus) this.onstatus(msg); }

  _fail(msg) {
    if (this.onstatus) this.onstatus('');
    if (this.onerror) this.onerror(new Error(msg));
    this.stop();
  }

  async play(url) {
    this.stop();
    if (!this.supported) { this._fail('MediaSource unsupported in this browser'); return; }
    this.connected = true;
    const MS = window.ManagedMediaSource || window.MediaSource;
    const ms = this.ms = new MS();
    this.video.src = URL.createObjectURL(ms);

    const wsUrl = String(url).replace(/^http/, 'ws');
    this._status('Connecting…');
    let ws;
    try {
      ws = this.ws = new WebSocket(wsUrl);
    } catch (e) {
      this._fail('cannot open WebSocket: ' + e.message);
      return;
    }
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      if (!this.connected) { try { ws.close(); } catch (e) { /* ignore */ } return; }
      this._status('Live');
      this.video.play().catch(() => {});
    };
    ws.onmessage = (ev) => this._onWs(ev, ms);
    ws.onerror = () => { if (this.connected) this._fail('WebSocket error'); };
    ws.onclose = () => { if (this.connected) this._fail('Stream closed'); };
  }

  _onWs(ev, ms) {
    if (!this.connected) return;
    if (typeof ev.data === 'string') {
      let msg = null;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (!msg) return;
      if (msg.type === 'mse' && msg.value) {
        this._ensureSource(ms, String(msg.value));
      } else if (msg.type === 'error') {
        this._fail('server: ' + String(msg.value || 'error'));
      }
      return;
    }
    if (ev.data instanceof ArrayBuffer) {
      if (!this.sb) {
        if (this.queue.length < 200) this.queue.push(ev.data);
      } else {
        this._enqueue(ev.data);
      }
    }
  }

  _ensureSource(ms, value) {
    if (this.sb) return;
    let mime = String(value);
    if (!/^video\/mp4/.test(mime)) mime = 'video/mp4; codecs="' + mime + '"';
    try {
      this.sb = ms.addSourceBuffer(mime);
      this.sb.mode = 'segments';
      this._evict = setInterval(() => this._evictOld(), 5000);
      const q = this.queue.splice(0);
      q.forEach(c => this._enqueue(c));
      this._status('Live');
    } catch (e) {
      this._fail('addSourceBuffer(' + mime + ') failed: ' + e.message);
    }
  }

  _enqueue(chunk) {
    if (!this.sb) return;
    this.queue.push(chunk);
    if (!this.sb.updating) this._drain();
  }

  _drain() {
    if (!this.sb || !this.queue.length || this.sb.updating) return;
    const chunk = this.queue.shift();
    try {
      this.sb.appendBuffer(chunk);
      this.sb.addEventListener('updateend', () => this._drain(), { once: true });
    } catch (e) {
      // QuotaExceeded/state hiccup - drop this chunk and continue draining.
      this._drain();
    }
  }

  _evictOld() {
    try {
      const v = this.video, sb = this.sb;
      if (!sb || sb.updating || !v || !v.buffered || !v.buffered.length) return;
      const n = v.buffered.length;
      const end = v.buffered.end(n - 1);
      const start = v.buffered.start(0);
      const keep = Math.max(2, v.currentTime - 5);
      if (end - start > 12 && keep > start && keep < end) sb.remove(start, keep);
    } catch (e) { /* ignore */ }
  }

  stop() {
    this.connected = false;
    if (this._evict) { clearInterval(this._evict); this._evict = null; }
    if (this.ws) {
      try {
        this.ws.onopen = this.ws.onmessage = this.ws.onerror = this.ws.onclose = null;
        this.ws.close();
      } catch (e) { /* ignore */ }
      this.ws = null;
    }
    try { if (this.ms && this.ms.readyState === 'open') this.ms.endOfStream(); } catch (e) { /* ignore */ }
    this.sb = null;
    this.queue = [];
    const v = this.video;
    if (v) {
      try { v.pause(); } catch (e) { /* ignore */ }
      try { v.removeAttribute('src'); v.load(); } catch (e) { /* ignore */ }
    }
    this.ms = null;
  }
}
