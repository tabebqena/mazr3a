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
import subprocess
import time
import urllib.error
import urllib.request

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif"}

DEFAULT_PROMPT = ("Describe this CCTV frame in one short sentence. Use the given "
                  "facts and add only what is plainly visible. Do not invent "
                  "places, names or activity.")


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
                 keep_loaded=True, port=8737, start_timeout=90.0, prompt=None):
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
        self.prompt = prompt or DEFAULT_PROMPT
        self._proc = None
        self._port = int(port)
        self._server_bin = _which([server_bin, "llama-server"])
        self._cli_bin = _which([cli_bin, "llama-mtmd-cli", "llama-mtmd"])
        if self.keep_loaded and self._server_bin:
            self._start_server(float(start_timeout))

    # -- server lifecycle ---------------------------------------------------
    def _start_server(self, timeout):
        cmd = [self._server_bin, "-m", self.model_file, "--mmproj", self.mmproj_file,
               "-c", str(self.ctx), "-t", str(self.n_threads),
               "--host", "127.0.0.1", "--port", str(self._port), "-ngl", "0"]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self._proc = None
            return
        deadline = time.monotonic() + max(1.0, timeout)
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._proc = None            # crashed - fall back to the CLI
                return
            if self._health():
                return
            time.sleep(0.5)

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

    # -- captioning ---------------------------------------------------------
    def caption(self, image_path, prompt=None):
        prompt = prompt or self.prompt
        started = time.monotonic()
        text = None
        if self._proc is not None and self._proc.poll() is None:
            text = self._caption_server(image_path, prompt)
        if text is None:
            text = self._caption_cli(image_path, prompt)
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
                obj = json.loads(resp.read().decode("utf-8", "replace"))
            return obj["choices"][0]["message"]["content"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
            return None

    def _caption_cli(self, image_path, prompt):
        if not self._cli_bin:
            return None
        cmd = [self._cli_bin, "-m", self.model_file, "--mmproj", self.mmproj_file,
               "--image", image_path, "-p", prompt, "-n", str(self.max_tokens),
               "-t", str(self.n_threads)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        # the CLI echoes the prompt back; keep the answer lines only
        lines = [ln.strip() for ln in (out.stdout or "").splitlines()]
        body = [ln for ln in lines if ln and not ln.startswith("llama_")
                and prompt.strip()[:24] not in ln]
        return " ".join(body).strip() or None


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
            port=int(getattr(settings, "llama_port", 8737) or 8737), prompt=prompt)
    raise RuntimeError("unknown MODEL_BACKEND {!r} (use llamacpp|openvino)".format(backend))
