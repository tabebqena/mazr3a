/* Minimal go2rtc MSE player (vanilla JS, no build).

go2rtc serves a live stream as MediaSource: fetch the MSE URL (Accept:
video/mp4), read the long-lived chunked body, and append each chunk to a
SourceBuffer created from the codecs advertised in the response Content-Type
(e.g. `video/mp4; codecs="avc1.64001F"`). H.264 remux only - the browser does
the decode. One connection at a time; stop() aborts it (which also lets go2rtc
release the camera pull).
*/
'use strict';

class G2MSEPlayer {
  constructor(video) {
    this.video = video;
    this.connected = false;
    this.abort = null;
    this.ms = null;
    this.sb = null;
    this._evict = null;
    this.onstatus = null;   // (msg) => void
    this.onerror = null;    // (err) => void
  }

  get supported() {
    return typeof window !== 'undefined' &&
      window.MediaSource && window.MediaSource.isTypeSupported('video/mp4');
  }

  _status(msg) {
    if (this.onstatus) this.onstatus(msg);
  }

  async play(url) {
    this.stop();
    if (!this.supported) {
      if (this.onerror) this.onerror(new Error('MediaSource unsupported in this browser'));
      return;
    }
    this.connected = true;
    const ms = this.ms = new MediaSource();
    this.video.src = URL.createObjectURL(ms);

    try {
      await new Promise((resolve, reject) => {
        ms.addEventListener('sourceopen', resolve, { once: true });
        ms.addEventListener('sourceerror', () => reject(new Error('MediaSource sourceerror')), { once: true });
      });
    } catch (e) {
      if (this.connected && this.onerror) this.onerror(e);
      this.stop();
      return;
    }
    if (!this.connected) return;

    const ctrl = this.abort = new AbortController();
    let resp;
    try {
      this._status('Connecting…');
      resp = await fetch(url, { signal: ctrl.signal, headers: { Accept: 'video/mp4' } });
    } catch (e) {
      if (this.connected) {
        if (this.onstatus) this.onstatus('');
        if (this.onerror) this.onerror(new Error('Stream fetch failed'));
      }
      this.stop();
      return;
    }
    if (!resp.ok) {
      if (this.onstatus) this.onstatus('');
      if (this.onerror) this.onerror(new Error('Stream HTTP ' + resp.status));
      this.stop();
      return;
    }

    let mime = 'video/mp4';
    const ct = resp.headers.get('Content-Type') || '';
    const m = ct.match(/codecs="([^"]+)"/);
    if (m) mime = 'video/mp4; codecs="' + m[1] + '"';
    try {
      this.sb = ms.addSourceBuffer(mime);
    } catch (e) {
      if (this.onstatus) this.onstatus('');
      if (this.onerror) this.onerror(new Error('Unsupported codec: ' + mime));
      this.stop();
      return;
    }

    this._status('Live');
    if (this.video.play) this.video.play().catch(() => {});
    this._evict = setInterval(() => this._evictOld(), 5000);

    const reader = resp.body.getReader();
    const pump = async () => {
      try {
        while (this.connected) {
          const { done, value } = await reader.read();
          if (done) break;
          if (!this.connected) break;
          await this._append(value);
        }
      } catch (e) {
        // AbortError on stop() is expected.
      } finally {
        this.stop();
      }
    };
    pump();
  }

  _append(chunk) {
    return new Promise((resolve) => {
      if (!this.sb) return resolve();
      if (this.sb.updating) {
        this.sb.addEventListener('updateend', () => resolve(this._append(chunk)), { once: true });
        return;
      }
      try {
        this.sb.appendBuffer(chunk);
        this.sb.addEventListener('updateend', () => resolve(), { once: true });
      } catch (e) {
        resolve(); // quota or state hiccup - skip this chunk
      }
    });
  }

  _evictOld() {
    try {
      const sb = this.sb, v = this.video;
      if (!sb || sb.updating || !v || !v.buffered || !v.buffered.length) return;
      const last = v.buffered.length - 1;
      const end = v.buffered.end(last);
      const keep = Math.max(2, v.currentTime - 5); // drop everything older than 5s back
      const start = v.buffered.start(0);
      if (end - start > 10 && keep > start && keep < end) {
        sb.remove(start, keep);
      }
    } catch (e) { /* ignore */ }
  }

  stop() {
    this.connected = false;
    if (this._evict) { clearInterval(this._evict); this._evict = null; }
    if (this.abort) { try { this.abort.abort(); } catch (e) {} this.abort = null; }
    try { if (this.ms && this.ms.readyState === 'open') this.ms.endOfStream(); } catch (e) {}
    this.sb = null;
    const v = this.video;
    if (v) {
      try { v.pause(); } catch (e) {}
      try { v.removeAttribute('src'); v.load(); } catch (e) {}
    }
    if (this.ms) { this.ms = null; }
  }
}
