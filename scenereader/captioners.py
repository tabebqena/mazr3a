"""captioners.py - the captioner backends behind one interface.

(Named `captioners`, not `models`, so it can never be confused with the repo's
top-level `models/` directory of model artifacts.)

Two backends, selected by MODEL_BACKEND (plans/event-scene-reader.md §8.1):

  llamacpp  DEFAULT. A small (<=500M) SmolVLM2 GGUF + mmproj, served by a
            RESIDENT `llama-server` child process. Chosen first because the goal
            is the smallest useful model.
  openvino  RETAINED. The already-downloaded Qwen2-VL-2B int4 OpenVINO IR in
            models/scene/ (openvino_genai.VLMPipeline on CPU). It is NOT deleted
            and NOT re-downloaded - flip MODEL_BACKEND to use it again.

RESIDENCY: both backends load ONCE and stay resident (`MODEL_KEEP_LOADED=true`,
the default). For a <=500M model the reload cost of load/unload outweighs the
RAM it would free, so unload-after-idle is deferred to an optimization
(`MODEL_KEEP_LOADED=false`, `IDLE_UNLOAD_S`).

Nothing here decides WHEN to caption - the service's idle governor does that.
"""
import base64
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif"}

DEFAULT_PROMPT = ("Describe this CCTV frame in one short sentence. Use the given "
                  "facts and add only what is plainly visible. Do not invent "
                  "places, names or activity.")

# llama.cpp prefixes its own log lines with a timestamp + level, e.g.
#   "0.00.495.942 I main: loading model: ..."   /  "0.00.499.568 E probe: ..."
# We filter by THIS pattern rather than passing --log-disable: that flag
# silences the generated text too (the completion goes through the same logging
# stream in this build), which is why the CLI exited 0 with EMPTY stdout.
_LOG_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+[IWED]\s")


def _which(names):
    """First existing executable among `names` (absolute paths accepted)."""
    for name in names:
        if not name:
            continue
        if os.path.isabs(name):
            if os.path.isfile(name) and os.access(name, os.X_OK):
                return name
        else:
            for directory in os.environ.get("PATH", "").split(os.pathsep):
                candidate = os.path.join(directory, name)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
    return None


class Captioner:
    """Common interface: `.caption(path, prompt) -> (text, latency_ms)`."""

    backend = "none"
    name = "none"
    device = "CPU"

    def caption(self, image_path, prompt=None):  # noqa: ARG002 - interface
        raise NotImplementedError

    def close(self):
        pass

    def alive(self):
        return True


# ---------------------------------------------------------------------------
# llama.cpp (small GGUF) - resident server, one-shot CLI fallback
# ---------------------------------------------------------------------------
class LlamaCppCaptioner(Captioner):
    """SmolVLM2 (or any mtmd-capable) GGUF served by a resident llama-server."""

    backend = "llamacpp"

    def __init__(self, model_file, mmproj_file, server_bin="llama-server",
                 cli_bin="llama-mtmd-cli", n_threads=2, ctx=4096, max_tokens=64,
                 keep_loaded=True, port=8737, start_timeout=180.0,
                 max_image_px=384, cli_timeout=300.0, prompt=None):
        if not model_file or not os.path.isfile(model_file):
            raise RuntimeError("MODEL_FILE not found: {!r} (run "
                               "dev_scripts/prep_scene_model_llamacpp.sh)".format(model_file))
        self.model_file = model_file
        self.mmproj_file = mmproj_file if mmproj_file and os.path.isfile(mmproj_file) else None
        if not self.mmproj_file:
            raise RuntimeError("MMPROJ_FILE not found: {!r} (the vision projector is "
                               "required for image input)".format(mmproj_file))
        self.name = os.path.basename(model_file)
        self.max_tokens = max(1, int(max_tokens))
        self.n_threads = max(1, int(n_threads))
        self.ctx = max(512, int(ctx))
        self.keep_loaded = bool(keep_loaded)
        # Cap the longest side handed to the vision encoder. Measured on the host:
        # encoding a 640x360 frame cost ~58 s of an ~87 s caption (two ~29 s mtmd
        # passes), while SmolVLM2's encoder works at 384 px - feeding it a smaller
        # image cuts the dominant cost with no loss of usable detail. 0 disables.
        self.max_image_px = max(0, int(max_image_px or 0))
        # Wall-clock ceiling for ONE one-shot CLI caption (VLM_TIMEOUT_S). The CLI
        # fallback is the slowest path (a full model load + encode + generate per
        # call), and the ONLY step that can look like a hang - so it is bounded,
        # reported, and the failure says how to fix it.
        self.cli_timeout = max(30.0, float(cli_timeout or 300.0))
        self.prompt = prompt or DEFAULT_PROMPT
        self._proc = None
        self._port = int(port)
        # WHY a caption came back empty: a missing runtime, a binary that cannot
        # find its shared libraries, a bad model file... Silence here is what
        # makes this class of failure so confusing, so the reason is kept and
        # surfaced by the service's --check.
        self.last_error = ""
        # Which path produced the last answer ("server"/"cli"), and the raw CLI
        # text, so --check can show what actually came back instead of just
        # "empty".
        self.last_path = ""
        self.last_raw = ""
        # Kept SEPARATELY from last_error (which describes the CLI), so a server
        # failure is never masked by the CLI that ran afterwards.
        self.server_error = ""
        self._logfh = None
        self._log_path = os.path.join(tempfile.gettempdir(), "llama-server.log")
        self._server_bin = _which([server_bin, "llama-server"])
        self._cli_bin = _which([cli_bin, "llama-mtmd-cli", "llama-mtmd"])
        if self.keep_loaded and self._server_bin:
            self._start_server(float(start_timeout))

    # -- server lifecycle ---------------------------------------------------
    def _start_server(self, timeout):
        # --jinja: apply the model's OWN chat template. SmolVLM2 is an INSTRUCT
        # model; without its template it answers an unframed prompt with an
        # immediate EOS, which surfaces as an empty caption after a full
        # generation run.
        cmd = [self._server_bin, "-m", self.model_file, "--mmproj", self.mmproj_file,
               "-c", str(self.ctx), "-t", str(self.n_threads), "--jinja",
               "--host", "127.0.0.1", "--port", str(self._port), "-ngl", "0"]
        try:
            # Keep the server's own output: when it exits immediately (most often
            # a missing/incompatible shared library) that text IS the diagnosis.
            self._logfh = open(self._log_path, "wb")
            self._proc = subprocess.Popen(
                cmd, stdout=self._logfh, stderr=subprocess.STDOUT)
        except OSError as exc:
            self._proc = None
            self.last_error = "cannot start {}: {}".format(self._server_bin, exc)
            return
        deadline = time.monotonic() + max(1.0, timeout)
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self.last_error = self._log_tail() or (
                    "llama-server exited with code {}".format(self._proc.returncode))
                self._proc = None            # crashed - fall back to the CLI
                return
            if self._health():
                return
            time.sleep(0.5)
        self.last_error = self._log_tail() or "llama-server did not become ready"

    def _log_tail(self, limit=600):
        """Last lines of the server log (its startup failure reason)."""
        try:
            with open(self._log_path, "rb") as fh:
                return fh.read()[-limit:].decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def _health(self):
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:{}/health".format(self._port), timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def alive(self):
        """True while SOMETHING can still caption (a live server or the CLI)."""
        return (self._proc is not None and self._proc.poll() is None) or bool(self._cli_bin)

    def close(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None
        if self._logfh is not None:
            try:
                self._logfh.close()
            except Exception:  # noqa: BLE001
                pass
            self._logfh = None

    # -- captioning ---------------------------------------------------------
    # Formats llama.cpp's BUILT-IN decoder handles on its own. Anything else
    # (notably WebP - which is what the scan prefers, the un-annotated
    # `-clean.webp`) makes mtmd reach for ffprobe/ffmpeg and fail:
    #   "failed to launch ffprobe" -> "failed to decode webp buffer"
    _NATIVE_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")

    def _prepare_image(self, image_path):
        """(path_to_use, is_temp): transcode if needed and cap the size.

        Pillow is already a dependency (the OpenVINO backend needs it), so this
        avoids shipping ffmpeg purely to decode WebP (mtmd shells out to
        ffprobe/ffmpeg for it and fails), and downscaling to the encoder's own
        working size is the single biggest CPU saving per caption.

        A native image that is already small enough is passed through untouched.
        """
        native = os.path.splitext(image_path)[1].lower() in self._NATIVE_IMAGE_EXT
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                frame = img.convert("RGB")
                width, height = frame.size
                longest = max(width, height)
                if native and (not self.max_image_px or longest <= self.max_image_px):
                    return image_path, False
                if self.max_image_px and longest > self.max_image_px:
                    scale = float(self.max_image_px) / float(longest)
                    # Image.LANCZOS is alive across the pinned Pillow range
                    # (>=9.1,<12) as an alias of Image.Resampling.LANCZOS; the
                    # stub lags, hence the ignore.
                    frame = frame.resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.LANCZOS)  # type: ignore[attr-defined]
                out = os.path.join(tempfile.gettempdir(),
                                   "scenereader-{}.jpg".format(os.getpid()))
                frame.save(out, "JPEG", quality=90)
                return out, True
        except Exception as exc:  # noqa: BLE001 - fall back and let the model speak
            self.last_error = "cannot prepare {}: {}".format(image_path, exc)
            return image_path, False

    def caption(self, image_path, prompt=None):
        prompt = prompt or self.prompt
        started = time.monotonic()
        prepared, is_temp = self._prepare_image(image_path)
        try:
            text = None
            if self._proc is not None and self._proc.poll() is None:
                text = self._caption_server(prepared, prompt)
                if text is not None:
                    self.last_path = "server"
            if text is None:
                text = self._caption_cli(prepared, prompt)
                self.last_path = "cli"
        finally:
            if is_temp:
                try:
                    os.remove(prepared)
                except OSError:
                    pass
        return (text or "").strip(), int((time.monotonic() - started) * 1000)

    def _caption_server(self, image_path, prompt):
        try:
            with open(image_path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return None
        mime = _MIME.get(os.path.splitext(image_path)[1].lower(), "image/jpeg")
        payload = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:{};base64,{}".format(mime, base64.b64encode(raw).decode())}},
            ]}],
        }
        req = urllib.request.Request(
            "http://127.0.0.1:{}/v1/chat/completions".format(self._port),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8", "replace")
            content = json.loads(raw)["choices"][0]["message"]["content"]
            self.server_error = ""
            return content
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                detail = ""
            self.server_error = "HTTP {}: {}".format(exc.code, detail.strip())
            return None
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            self.server_error = "{}: {}".format(type(exc).__name__, exc)
            return None

    def _caption_cli(self, image_path, prompt):
        if not self._cli_bin:
            return None
        cmd = [self._cli_bin, "-m", self.model_file, "--mmproj", self.mmproj_file,
               "--image", image_path, "-p", prompt, "-n", str(self.max_tokens),
               "-t", str(self.n_threads), "--jinja"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=self.cli_timeout)
        except subprocess.TimeoutExpired:
            self.last_error = ("{} timed out after {:g}s on {} threads - raise "
                               "VLM_TIMEOUT_S (or lower VLM_MAX_IMAGE_PX) and "
                               "retry".format(os.path.basename(self._cli_bin),
                                              self.cli_timeout, self.n_threads))
            return None
        except (OSError, subprocess.SubprocessError) as exc:
            self.last_error = "{} failed: {}".format(self._cli_bin, exc)
            return None
        if out.returncode != 0:
            detail = ((out.stderr or "") + (out.stdout or "")).strip()
            self.last_error = "{} exited {}: {}".format(
                os.path.basename(self._cli_bin), out.returncode, detail[-400:])
            return None
        # Keep the raw text for --check, drop llama.cpp's own log lines by
        # PATTERN (not --log-disable, which also hides the answer), then drop the
        # echoed prompt.
        self.last_raw = ((out.stdout or "") + "\n---stderr---\n"
                         + (out.stderr or ""))[-800:]
        # Take everything AFTER the LAST log-prefixed line. Line-by-line filtering
        # is not enough: --jinja also dumps a multi-line "chat template example"
        # whose continuation lines have no timestamp, and those would leak into
        # the caption. The generated answer is always printed after the final
        # "mtmd batch encoding done" line, so the tail is exactly what we want.
        body = []
        for block in ((out.stdout or ""), (out.stderr or "")):
            lines = block.splitlines()
            last_log = -1
            for idx, ln in enumerate(lines):
                if _LOG_LINE_RE.match(ln):
                    last_log = idx
            body.extend(lines[last_log + 1:])
        head = prompt.strip()[:24]
        text = " ".join(ln.strip() for ln in body
                        if ln.strip() and not (head and head in ln)).strip()
        echo = prompt.strip()
        if echo and text.startswith(echo):
            text = text[len(echo):].strip(" :\n\t-")
        return text or None


# ---------------------------------------------------------------------------
# OpenVINO GenAI (the RETAINED 2B IR)
# ---------------------------------------------------------------------------
class OpenVinoCaptioner(Captioner):
    """Qwen2-VL-2B int4 via openvino_genai.VLMPipeline, loaded once and kept.

    Kept so the previously-downloaded export keeps working; the architecture list
    VLMPipeline implements is closed (llava/qwen2_vl/qwen2_5_vl/gemma3/minicpm/
    phi3_v/phi4mm) which is why it cannot go below ~2B.
    """

    backend = "openvino"

    def __init__(self, model_dir, device="CPU", threads=2, max_tokens=64, prompt=None):
        # Validate the export FIRST so a missing/moved model reports the real
        # problem instead of an import error.
        model_dir = str(model_dir or "").rstrip("/")
        if not os.path.isfile(os.path.join(model_dir, "config.json")):
            raise RuntimeError("no OpenVINO VLM at {!r} (config.json missing) - the "
                               "retained 2B IR must still be present".format(model_dir))
        try:
            import openvino_genai as ov_genai  # type: ignore[import-not-found]
        except ImportError as exc:  # image without the retained runtime
            raise RuntimeError(
                "openvino-genai is not available in this image: {}".format(exc))

        self.model_dir = model_dir
        self.name = os.path.basename(model_dir) or model_dir
        self.device = str(device or "CPU").upper()
        self.max_tokens = max(1, int(max_tokens))
        self.prompt = prompt or DEFAULT_PROMPT
        self._threads = max(0, int(threads))
        props = {}
        if self.device.startswith("CPU") and self._threads > 0:
            props["INFERENCE_NUM_THREADS"] = str(self._threads)
        self.pipe = None
        try:
            self.pipe = ov_genai.VLMPipeline(model_dir, self.device, props)
        except TypeError:
            self.pipe = ov_genai.VLMPipeline(model_dir, self.device)

    def caption(self, image_path, prompt=None):
        import numpy as np
        import openvino as ov
        import openvino_genai as ov_genai  # type: ignore[import-not-found]
        from PIL import Image

        prompt = prompt or self.prompt
        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("captioner is closed")
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        tensor = ov.Tensor(arr)
        gen = ov_genai.GenerationConfig()
        gen.max_new_tokens = self.max_tokens
        started = time.monotonic()
        res = None
        for kwargs in ({"image": tensor, "generation_config": gen},
                       {"images": tensor, "generation_config": gen},
                       {"image": tensor, "max_new_tokens": self.max_tokens},
                       {"images": tensor, "max_new_tokens": self.max_tokens}):
            try:
                res = pipe.generate(prompt, **kwargs)
                break
            except TypeError:
                continue
        if res is None:
            res = pipe.generate(prompt, tensor, gen)
        latency = int((time.monotonic() - started) * 1000)
        texts = getattr(res, "texts", None)
        text = str(texts[0]).strip() if texts else str(res).strip()
        return text, latency

    def close(self):
        self.pipe = None


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_captioner(settings):
    """Build the configured backend. Raises RuntimeError with a clear message."""
    backend = (getattr(settings, "model_backend", "llamacpp") or "llamacpp").lower()
    prompt = getattr(settings, "vlm_prompt", None) or DEFAULT_PROMPT
    max_tokens = int(getattr(settings, "vlm_max_tokens", 64) or 64)
    threads = int(getattr(settings, "vlm_n_threads", 2) or 2)
    if backend == "openvino":
        return OpenVinoCaptioner(getattr(settings, "openvino_dir", "/models/scene"),
                                 device=getattr(settings, "openvino_device", "CPU"),
                                 threads=int(getattr(settings, "openvino_threads", threads)),
                                 max_tokens=max_tokens, prompt=prompt)
    if backend == "llamacpp":
        return LlamaCppCaptioner(
            getattr(settings, "model_file", ""), getattr(settings, "mmproj_file", ""),
            server_bin=getattr(settings, "llama_server_bin", "llama-server"),
            cli_bin=getattr(settings, "llama_cli_bin", "llama-mtmd-cli"),
            n_threads=threads, ctx=int(getattr(settings, "vlm_ctx", 4096) or 4096),
            max_tokens=max_tokens,
            keep_loaded=bool(getattr(settings, "model_keep_loaded", True)),
            port=int(getattr(settings, "llama_port", 8737) or 8737),
            max_image_px=int(getattr(settings, "vlm_max_image_px", 384) or 384),
            cli_timeout=float(getattr(settings, "vlm_timeout_s", 300) or 300),
            prompt=prompt)
    raise RuntimeError("unknown MODEL_BACKEND {!r} (use llamacpp|openvino)".format(backend))
