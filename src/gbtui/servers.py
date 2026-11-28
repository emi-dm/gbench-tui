"""Endpoints: probing an OpenAI-compatible server and, for a local GGUF (+ optional LoRA), starting / stopping `llama-server`.

llama-server runs on the host (it needs Metal / CUDA); gbench and Harbor reach it from their containers. Its memory is
checked against what the host has free before it starts (base weights + LoRA weights + KV cache estimate), and it uses
`--lora` / `--lora-scaled` when a harness adapter GGUF is configured.
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
# llama-server for a local GGUF (+ optional LoRA adapter)
# ------------------------------------------------------------------------------------------------
def _servers_dir() -> Path:
    d = config_dir() / "servers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_file(ep: Endpoint) -> Path:
    return _servers_dir() / f"{_safe_name(ep.name)}.json"


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


def _read_state(path: Path) -> dict | None:
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
        return st if isinstance(st, dict) else None
    except (OSError, ValueError):
        return None


def server_pid(ep: Endpoint) -> int | None:
    st = _read_state(_state_file(ep))
    pid = st.get("pid") if st else None
    if isinstance(pid, int) and pid > 0 and _alive(pid):
        return pid
    _state_file(ep).unlink(missing_ok=True)
    return None


def _kill_pid(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    for _ in range(30):                             # up to 1.5s graceful exit (snappy quit)
        if not _alive(pid):
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        _alive(pid)


LOCAL_DISCOVERY_PORTS: tuple[tuple[str, int, str], ...] = (
    ("ollama", 11434, "http://127.0.0.1:11434/v1"),
    ("lm-studio", 1234, "http://127.0.0.1:1234/v1"),
    ("llama-server", 8080, "http://127.0.0.1:8080/v1"),
    ("llama-server-8081", 8081, "http://127.0.0.1:8081/v1"),
    ("vllm", 8000, "http://127.0.0.1:8000/v1"),
)


def discover_local(timeout: float = 0.7) -> list[dict]:
    """Probes standard local OpenAI-compatible ports (Ollama, LM Studio, llama-server, vLLM) in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    def _check(item):
        kind, port, url = item
        pr = probe(Endpoint(name=kind, base_url=url), timeout=timeout)
        if pr["ok"]:
            return {"kind": kind, "port": port, "base_url": url, "models": pr["models"], "latency_ms": pr["latency_ms"]}
        return None

    with ThreadPoolExecutor(max_workers=len(LOCAL_DISCOVERY_PORTS)) as ex:
        return [r for r in ex.map(_check, LOCAL_DISCOVERY_PORTS) if r is not None]


def stop_all_servers(endpoints: list) -> list[str]:
    """Stops all running gbtui-managed llama-servers in parallel for fast exit."""
    from concurrent.futures import ThreadPoolExecutor

    live = started_servers(endpoints)
    if not live:
        return []
    with ThreadPoolExecutor(max_workers=max(1, len(live))) as ex:
        list(ex.map(stop_server, live))
    return [e.name for e in live]


def stop_port_conflicts(ep: Endpoint, on_line=None) -> list[str]:
    """Stops any other gbtui-managed llama-server already bound to ep.port (e.g. switching base <-> LoRA)."""
    say = on_line or (lambda _m: None)
    stopped: list[str] = []
    mine = _state_file(ep)
    for sf in _servers_dir().glob("*.json"):
        if sf == mine:
            continue
        st = _read_state(sf)
        if not st:
            continue
        pid = st.get("pid")
        port = st.get("port")
        if port == ep.port and isinstance(pid, int) and pid > 0 and _alive(pid):
            say(f"stopping previous llama-server on port {ep.port} ({sf.stem}, pid {pid})…")
            _kill_pid(pid)
            sf.unlink(missing_ok=True)
            stopped.append(sf.stem)
        elif not (isinstance(pid, int) and pid > 0 and _alive(pid)):
            sf.unlink(missing_ok=True)
    return stopped


def estimate_bytes(ep: Endpoint) -> int:
    """Base weights + LoRA adapter weights + KV cache estimate + compute buffers."""
    try:
        weights = Path(ep.gguf).expanduser().stat().st_size if ep.gguf else 0
    except OSError:
        weights = 0
    try:
        lora_w = Path(ep.lora).expanduser().stat().st_size if ep.lora else 0
    except OSError:
        lora_w = 0
    return int(weights * 1.05 + lora_w * 1.15 + max(0, ep.ctx) * 0.06 * 1024 ** 2 + 600 * 1024 ** 2)


def llama_cmd(ep: Endpoint) -> list[str]:
    """Exact `llama-server` argv for `ep` (includes `--lora` / `--lora-scaled` when configured)."""
    gguf = Path(ep.gguf).expanduser()
    extra = shlex.split(ep.extra_args or "")
    cmd = ["llama-server", "-m", str(gguf), "--alias", ep.model or gguf.stem, "--host", "127.0.0.1",
           "--port", str(ep.port), "-c", str(max(512, ep.ctx)), "-np", str(max(1, ep.parallel)), "--jinja", "-ngl", "99",
           "--no-webui"]
    if ep.lora:
        lora_p = str(Path(ep.lora).expanduser())
        scale = float(ep.lora_scale if ep.lora_scale is not None else 1.0)
        if abs(scale - 1.0) > 1e-6:
            cmd += ["--lora-scaled", lora_p, f"{scale:g}"]
        else:
            cmd += ["--lora", lora_p]
    cmd += extra
    return cmd


def _config_matches_running(ep: Endpoint) -> bool:
    st = _read_state(_state_file(ep))
    if not st:
        return False
    try:
        want = llama_cmd(ep)
    except ValueError:
        return False
    return st.get("cmd") == want


def start_server(ep: Endpoint, min_free_gb: float, wait: float = 120, on_line=None) -> str | None:
    """Starts llama-server for the endpoint's GGUF (+ optional LoRA). None on success, else the reason."""
    say = on_line or (lambda _m: None)
    if not ep.gguf:
        return "this endpoint has no GGUF file: start its server yourself"
    gguf = Path(ep.gguf).expanduser()
    if not gguf.is_file():
        return f"GGUF not found: {gguf}"
    lora_p: Path | None = None
    if ep.lora:
        lora_p = Path(ep.lora).expanduser()
        if not lora_p.is_file():
            return f"LoRA GGUF not found: {lora_p}"
    if not llama_installed():
        return "llama-server not found (brew install llama.cpp)"
    if server_pid(ep):
        if _config_matches_running(ep) and probe(ep, timeout=1.5)["ok"]:
            return None
        say(f"restarting llama-server for {ep.name} (config changed)…")
        stop_server(ep)
    stop_port_conflicts(ep, on_line=say)
    if probe(ep, timeout=1.5)["ok"]:
        return None                                  # external server already serves that port
    need = estimate_bytes(ep)
    why = sysmem.check(need, min_free_gb)
    if why:
        return why
    try:
        cmd = llama_cmd(ep)
    except ValueError as e:
        return f"invalid llama-server extra args: {e}"
    if ep.ctx_warning:
        say("⚠ " + ep.ctx_warning)
    lora_note = (f" + lora {lora_p.name}" + (f" (scale {ep.lora_scale:g})" if abs(ep.lora_scale - 1.0) > 1e-6 else "")
                 if lora_p else "")
    say(f"llama-server {gguf.name}{lora_note} · ctx {ep.ctx} · {ep.parallel} slots · ~{sysmem.fmt(need)}")
    with open(log_file(ep), "ab") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    _state_file(ep).write_text(json.dumps({
        "pid": p.pid, "cmd": cmd, "port": ep.port, "gguf": str(gguf),
        "lora": str(lora_p) if lora_p else "", "lora_scale": float(ep.lora_scale),
        "started": time.time(),
    }), encoding="utf-8")
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
    _kill_pid(pid)
    _state_file(ep).unlink(missing_ok=True)
    return True


def started_servers(endpoints: list) -> list:
    return [e for e in endpoints if server_pid(e)]
