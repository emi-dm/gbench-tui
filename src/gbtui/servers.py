"""Endpoints: probing an OpenAI-compatible server and, for a local GGUF, starting / stopping `llama-server`.

llama-server runs on the host (it needs Metal / CUDA); gbench reaches it from its sandbox. Its memory is checked against
what the host has free before it starts (weights + an estimate of the KV cache), and it gets a modest context and few
slots by default, so a benchmark never pushes the machine into swap.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import sysmem
from .config import Endpoint, config_dir


def _key(ep: Endpoint) -> str:
    return os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name or "")).strip(".") or "endpoint"


def probe(ep: Endpoint, timeout: float = 4.0) -> dict:
    """{'ok': bool, 'models': [...], 'error': str, 'latency_ms': int}"""
    url = ep.base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_key(ep)}"} if _key(ep) else {})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode(errors="replace") or "{}")
        items = data.get("data", []) if isinstance(data, dict) else []
        ids = [m.get("id") for m in (items if isinstance(items, list) else []) if isinstance(m, dict) and m.get("id")]
        return {"ok": True, "models": ids, "error": "", "latency_ms": int((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        return {"ok": False, "models": [], "error": f"HTTP {e.code} {e.reason}", "latency_ms": 0}
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, TypeError, AttributeError) as e:
        reason = getattr(e, "reason", e)
        return {"ok": False, "models": [], "error": str(reason)[:120], "latency_ms": 0}


def chat_once(ep: Endpoint, prompt: str = "Say hello in five words.", timeout: float = 60,
              model: str | None = None) -> dict:
    """One short chat completion: checks the endpoint really answers (not only lists models)."""
    body = json.dumps({"model": model or ep.model or "default", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 48, "temperature": 0}).encode()
    headers = {"Content-Type": "application/json"}
    if _key(ep):
        headers["Authorization"] = f"Bearer {_key(ep)}"
    req = urllib.request.Request(ep.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode(errors="replace"))
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            return {"ok": False, "text": "unexpected response format (missing choices)", "seconds": 0}
        msg = data["choices"][0].get("message") or {}
        return {"ok": True, "text": str(msg.get("content") or msg.get("reasoning_content") or "").strip(),
                "seconds": round(time.time() - t0, 2), "usage": data.get("usage", {})}
    except urllib.error.HTTPError as e:
        return {"ok": False, "text": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}", "seconds": 0}
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
        return {"ok": False, "text": str(e)[:200], "seconds": 0}


# ------------------------------------------------------------------------------------------------
# llama-server for a local GGUF
# ------------------------------------------------------------------------------------------------
def _state_file(ep: Endpoint) -> Path:
    d = config_dir() / "servers"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_safe_name(ep.name)}.json"


def log_file(ep: Endpoint) -> Path:
    d = config_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"llama-server-{_safe_name(ep.name)}.log"


def llama_installed() -> bool:
    return shutil.which("llama-server") is not None


def _alive(pid: int) -> bool:
    # Reap if it's our own exited child so zombie processes don't look alive to os.kill(pid, 0)
    with contextlib.suppress(ChildProcessError, OSError):
        wpid, _status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def server_pid(ep: Endpoint) -> int | None:
    try:
        st = json.loads(_state_file(ep).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = st.get("pid") if isinstance(st, dict) else None
    if isinstance(pid, int) and pid > 0 and _alive(pid):
        return pid
    _state_file(ep).unlink(missing_ok=True)
    return None


def estimate_bytes(ep: Endpoint) -> int:
    """Weights (file size) + KV cache estimate + compute buffers. A rough upper bound for small models: ~0.06 MB per
    token of context for 1–4B models with a hybrid / GQA cache, plus 600 MB of buffers."""
    try:
        weights = Path(ep.gguf).expanduser().stat().st_size
    except OSError:
        weights = 0
    return int(weights * 1.05 + max(0, ep.ctx) * 0.06 * 1024 ** 2 + 600 * 1024 ** 2)


def start_server(ep: Endpoint, min_free_gb: float, wait: float = 120, on_line=None) -> str | None:
    """Starts llama-server for the endpoint's GGUF. None on success, else the reason."""
    say = on_line or (lambda _m: None)
    if not ep.gguf:
        return "this endpoint has no GGUF file: start its server yourself"
    gguf = Path(ep.gguf).expanduser()
    if not gguf.is_file():
        return f"GGUF not found: {gguf}"
    if not llama_installed():
        return "llama-server not found (brew install llama.cpp)"
    if server_pid(ep) and probe(ep, timeout=1.5)["ok"]:
        return None
    if probe(ep, timeout=1.5)["ok"]:
        return None                                  # something already serves that port
    need = estimate_bytes(ep)
    why = sysmem.check(need, min_free_gb)
    if why:
        return why
    try:
        extra = shlex.split(ep.extra_args or "")
    except ValueError as e:
        return f"invalid llama-server extra args: {e}"
    cmd = ["llama-server", "-m", str(gguf), "--alias", ep.model or gguf.stem, "--host", "127.0.0.1",
           "--port", str(ep.port), "-c", str(max(512, ep.ctx)), "-np", str(max(1, ep.parallel)), "--jinja", "-ngl", "99",
           "--no-webui", *extra]
    say(f"llama-server {gguf.name} · ctx {ep.ctx} · {ep.parallel} slots · ~{sysmem.fmt(need)}")
    with open(log_file(ep), "ab") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    _state_file(ep).write_text(json.dumps({"pid": p.pid, "cmd": cmd, "started": time.time()}), encoding="utf-8")
    t0 = time.time()
    while time.time() - t0 < wait:
        if p.poll() is not None:
            _state_file(ep).unlink(missing_ok=True)
            tail = log_file(ep).read_text(errors="replace").splitlines()[-4:]
            return "llama-server exited: " + " | ".join(tail)[-300:]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ep.port}/health", timeout=2) as r:
                if r.status == 200:
                    return None
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            pass
        time.sleep(1)
    stop_server(ep)
    return "llama-server did not become ready in time (see its log)"


def stop_server(ep: Endpoint) -> bool:
    pid = server_pid(ep)
    if not pid:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        _alive(pid)
    _state_file(ep).unlink(missing_ok=True)
    return True


def started_servers(endpoints: list) -> list:
    return [e for e in endpoints if server_pid(e)]
