"""A run: gbench's own CLI, inside a fresh Docker Sandbox, against one endpoint.

The command line is plain gbench (`gbench --golden-only|--evals-only|--personal --remote-endpoint … --models …
--tokenizer …`): gbench-tui only chooses the flags, starts the sandbox, streams the output and files the results.

Results: gbench writes to <workspace>/runs/<id>/ (~/.cache/gbench-tui/work, the sandbox's workspace); when the run ends its
`<timestamp>/` directory is moved to the results directory as `<id>/`, with `gbtui.json` (what was asked, how it
ended) next to gbench's own metadata.json / summary.json / eval_*.json.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import sandbox, servers, sysmem
from .config import Endpoint, Settings, config_dir

TIERS = {
    "golden": "Golden Set: 16 deterministic capability smoke tests",
    "evals": "Academic eval suites (pick suites; stratified subset with --eval-limit)",
    "perf": "Performance on this device (--personal: single-stream decode tok/s and TTFT)",
    "preflight": "Preflight: every selected suite's prerequisites, no model request",
}


@dataclass
class RunSpec:
    endpoint: Endpoint
    tier: str = "golden"
    evals: list = field(default_factory=list)
    golden_tasks: list = field(default_factory=list)
    eval_limit: int = 20
    max_output_tokens: int = 4096
    thinking: bool = False
    temperature: str = ""
    concurrency: int = 2
    suite_timeout: int = 3600
    perf_mode: str = "personal"

    def run_id(self) -> str:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = self.endpoint.name
        q = self.endpoint.effective_quant
        if q and q.lower() not in tag.lower().replace("-", "_"):
            tag = f"{tag}-{q}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)[:46]
        return f"{ts}-{safe}-{self.tier}"


def gbench_args(spec: RunSpec, results_dir: str) -> list[str]:
    """The gbench command line for a spec (the URL as the sandbox sees it)."""
    ep = spec.endpoint
    a = ["gbench", "--remote-endpoint", sandbox.sandbox_url(ep.base_url), "--results-dir", results_dir]
    if ep.model:
        a += ["--models", ep.model]
    if ep.tokenizer:
        a += ["--tokenizer", ep.tokenizer]
    if spec.tier == "golden":
        a += ["--golden-only"]
        if spec.golden_tasks:
            a += ["--golden-tasks", *spec.golden_tasks]
    elif spec.tier in ("evals", "preflight"):
        a += ["--evals-only", "--evals", *(spec.evals or ["gsm8k"])]
        if spec.tier == "preflight":
            a += ["--preflight", "--max-output-tokens", str(spec.max_output_tokens)]
        else:
            a += ["--max-output-tokens", str(spec.max_output_tokens)]
            if spec.eval_limit:
                a += ["--eval-limit", str(spec.eval_limit)]
            if spec.concurrency:
                a += ["--eval-concurrency", str(spec.concurrency)]
            if spec.suite_timeout:
                a += ["--suite-timeout", str(spec.suite_timeout)]
    elif spec.tier == "perf":
        if spec.perf_mode == "personal":
            a += ["--personal"]
            # a GGUF served by llama-server (no mmproj) is text-only; llama-server answers gbench's image probe
            # with a 500, which gbench reads as "inconclusive" and then fails the multimodal pillar
            if spec.endpoint.gguf:
                a += ["--text-only"]
        else:
            a += ["--serving-only", "--preset", "quick", "--text-only", "--no-stress-test"]
    if spec.tier != "perf":
        if spec.thinking:
            a += ["--eval-thinking"]
        if str(spec.temperature).strip():
            a += ["--temperature", str(spec.temperature).strip()]
    return a


def command_preview(spec: RunSpec) -> str:
    return " ".join(gbench_args(spec, f"{sandbox.work_dir()}/runs/<id>"))


# ------------------------------------------------------------------------------------------------
# live state of the run in progress (one at a time: protects the machine)
# ------------------------------------------------------------------------------------------------
@dataclass
class Live:
    run_id: str = ""
    spec: RunSpec | None = None
    lines: list = field(default_factory=list)
    dropped: int = 0               # total trimmed lines so readers keep stable indices
    phase: str = "idle"            # idle | server | sandbox | running | collecting | done | failed | cancelled
    started: float = 0.0
    finished: float = 0.0
    rc: int | None = None
    progress: tuple = (0, 0)       # (done, total) from the last tqdm / golden line
    current: str = ""              # suite / task being measured
    sandbox: str = ""
    result_dir: str = ""
    error: str = ""
    proc: subprocess.Popen | None = None
    cancel: bool = False
    _golden_seen: set = field(default_factory=set)
    _thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.phase in ("server", "sandbox", "running", "collecting")

    def add(self, line: str) -> None:
        with _LOCK:
            self.lines.append(line)
            if len(self.lines) > 4000:
                del self.lines[:1000]
                self.dropped += 1000


LIVE = Live()
_LOCK = threading.Lock()
_TQDM = re.compile(r"(?:(?P<desc>[^\r\n|]{0,60}):\s*)?\s*\d+%\|[^|]*\|\s*(?P<done>\d+)/(?P<total>\d+)")
_SUITE = re.compile(r"(?:Running|Evaluating|Starting)\s+(?:eval(?:uation)?\s+suite\s+)?['\"]?([a-z][a-z0-9_]{2,})['\"]?", re.I)
_SUITE_NOISE = {"the", "for", "with", "all", "and", "from", "model", "models", "benchmark", "benchmarks",
                "evaluation", "eval", "evals", "server", "sandbox", "tasks", "prompts", "golden", "serving"}
_GOLDEN_TOTAL = re.compile(r"Running Golden Set evaluation .*?\((\d+)\s+tasks?\)", re.I)
_GOLDEN_CASE = re.compile(r"Golden case ['\"]([a-z0-9_-]+)['\"]", re.I)


def snapshot() -> dict:
    """Thread-safe snapshot of LIVE state for UI/CLI readers."""
    with _LOCK:
        return {
            "run_id": LIVE.run_id, "spec": LIVE.spec, "lines": list(LIVE.lines), "dropped": LIVE.dropped,
            "phase": LIVE.phase, "active": LIVE.active, "started": LIVE.started, "finished": LIVE.finished,
            "rc": LIVE.rc, "progress": LIVE.progress, "current": LIVE.current, "sandbox": LIVE.sandbox,
            "result_dir": LIVE.result_dir, "error": LIVE.error,
        }


def wait(timeout: float | None = None) -> None:
    """Wait for the active background run thread (including sandbox cleanup) to finish."""
    t = LIVE._thread
    if t and t.is_alive():
        t.join(timeout=timeout)


def lock_file() -> Path:
    return config_dir() / "run.lock"


def other_run() -> dict | None:
    """A run started by another gbench-tui process that is still alive."""
    try:
        st = json.loads(lock_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = st.get("pid") if isinstance(st, dict) else None
    if pid and pid != os.getpid():
        try:
            os.kill(pid, 0)
            return st
        except ProcessLookupError:
            lock_file().unlink(missing_ok=True)
        except PermissionError:
            return st
    return None


def start(spec: RunSpec, settings: Settings) -> str | None:
    """Starts the run in a background thread. None when started, else why not."""
    with _LOCK:
        if LIVE.active or (LIVE._thread and LIVE._thread.is_alive()):
            return "a run is already in progress (one at a time, to keep the machine responsive)"
        if other_run():
            return "another gbench-tui is running a benchmark (one at a time)"
        rid = spec.run_id()
        started = time.time()
        try:
            lock_file().write_text(json.dumps({"pid": os.getpid(), "run_id": rid, "started": started}), encoding="utf-8")
        except OSError as e:
            return f"could not acquire run lock: {e}"
        LIVE.__init__()
        LIVE.spec, LIVE.run_id, LIVE.phase, LIVE.started = spec, rid, "server", started
        t = threading.Thread(target=_run, args=(spec, settings), daemon=True)
        LIVE._thread = t
    t.start()
    return None


def cancel() -> None:
    with _LOCK:
        LIVE.cancel = True
        p = LIVE.proc
        box = LIVE.sandbox
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGINT)
        except OSError:
            pass
    if box:                                         # stops gbench and everything it started, at once
        threading.Thread(target=sandbox.destroy, args=(box,), daemon=True).start()


def _say(msg: str) -> None:
    LIVE.add(msg)


def _handle_line(line: str, log) -> None:
    log.write(line + "\n")
    _parse_progress(line)
    if not _TQDM.search(line):
        _say(line)
    else:
        with _LOCK:
            if LIVE.lines and _TQDM.search(LIVE.lines[-1]):
                LIVE.lines[-1] = line
            else:
                LIVE.lines.append(line)
                if len(LIVE.lines) > 4000:
                    del LIVE.lines[:1000]
                    LIVE.dropped += 1000


def _run(spec: RunSpec, settings: Settings) -> None:
    ep = spec.endpoint
    box = ""
    work: Path | None = None
    final_phase = "failed"
    try:
        work = sandbox.work_dir() / "runs" / LIVE.run_id
        work.mkdir(parents=True, exist_ok=True)
        # 1. the endpoint (start llama-server for a GGUF)
        if ep.gguf and not servers.probe(ep, timeout=1.5)["ok"]:
            _say("▸ starting llama-server…")
            why = servers.start_server(ep, settings.min_free_gb, on_line=_say)
            if why:
                raise RuntimeError(why)
        pr = servers.probe(ep)
        if not pr["ok"]:
            raise RuntimeError(f"endpoint {ep.base_url} not reachable: {pr['error']}")
        _say(f"✔ endpoint {ep.base_url} serves: {', '.join(pr['models'][:4])}")
        if LIVE.cancel:
            raise InterruptedError
        # 2. the sandbox, if the host has room for it
        LIVE.phase = "sandbox"
        why = sysmem.check(settings.sandbox_memory_gb * sysmem.GB, settings.min_free_gb)
        if why:
            raise RuntimeError(why)
        box = sandbox.create(settings.sandbox_memory_gb, settings.cpus(), sandbox.allow_for(ep.base_url), _say)
        LIVE.sandbox = box
        if LIVE.cancel:
            raise InterruptedError
        # 3. gbench
        LIVE.phase = "running"
        wdir = str(sandbox.work_dir())
        args = gbench_args(spec, f"{wdir}/runs/{LIVE.run_id}")
        _say("$ " + " ".join(args))
        env = {"HF_HOME": f"{wdir}/hf", "NLTK_DATA": "/opt/nltk_data", "PYTHONUNBUFFERED": "1",
               "TQDM_MININTERVAL": "2", "TOKENIZERS_PARALLELISM": "false", "COLUMNS": "120"}
        key = os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""
        if key:
            env["VLLM_API_KEY"] = key
        cmd = sandbox.exec_argv(box, ["sh", "-c", f"cd {wdir} && exec " + " ".join(_q(x) for x in args)], env=env)
        with open(work / "gbench.log", "w", encoding="utf-8") as log:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=sandbox._env(), bufsize=0)
            LIVE.proc = p
            buf = b""
            while True:
                chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                for raw in parts:
                    line = raw.decode(errors="replace").rstrip()
                    if line:
                        _handle_line(line, log)
            if buf:
                tail = buf.decode(errors="replace").rstrip()
                if tail:
                    _handle_line(tail, log)
            LIVE.rc = p.wait()
        if LIVE.cancel:
            raise InterruptedError
        LIVE.phase = "collecting"
        LIVE.result_dir = str(_collect(work, spec, settings))
        final_phase = "done" if LIVE.rc in (0, 1) else "failed"
        verdict = {0: "every pillar produced its result", 1: "the model failed some Golden Set cases",
                   2: "a harness error (see the log): not a verdict on the model"}.get(LIVE.rc, f"exit code {LIVE.rc}")
        _say(f"{'✔' if LIVE.rc in (0, 1) else '✗'} gbench finished: {verdict}")
    except InterruptedError:
        final_phase = "cancelled"
        LIVE.phase = "collecting"
        _say("■ cancelled")
        if work and work.is_dir() and any(work.iterdir()):
            LIVE.result_dir = str(_collect(work, spec, settings, cancelled=True))
    except Exception as e:                           # noqa: BLE001 - shown in the live view
        final_phase = "failed"
        LIVE.error = str(e)
        LIVE.phase = "collecting"
        _say(f"✗ {e}")
        if work and work.is_dir() and any(work.iterdir()):
            LIVE.result_dir = str(_collect(work, spec, settings))
    finally:
        if box:
            _say(f"removing sandbox {box}…")
            sandbox.destroy(box)
        try:
            from . import leaderboard
            leaderboard.write(settings)
        except Exception:                             # noqa: BLE001
            pass
        lock_file().unlink(missing_ok=True)
        with _LOCK:
            LIVE.finished = time.time()
            LIVE.proc = None
            LIVE.phase = final_phase


def _q(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./:=@,+-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def _parse_progress(line: str) -> None:
    m = _TQDM.search(line)
    if m:
        done, total = int(m.group("done")), int(m.group("total"))
        if total > 0:
            done = min(done, total)
        LIVE.progress = (done, total)
        if m.group("desc"):
            LIVE.current = m.group("desc").strip()[-40:]
        return
    m = _GOLDEN_TOTAL.search(line)
    if m:
        LIVE.current = "Golden Set"
        LIVE._golden_seen = set()
        LIVE.progress = (0, int(m.group(1)))
        return
    m = _GOLDEN_CASE.search(line)
    if m:
        cid = m.group(1)
        LIVE.current = cid
        _, t = LIVE.progress
        if t:
            LIVE._golden_seen.add(cid)
            LIVE.progress = (min(t, len(LIVE._golden_seen)), t)
        return
    if "Running Golden Set" in line:
        LIVE.current = "Golden Set"
        return
    m = _SUITE.search(line)
    if m:
        word = m.group(1)
        if word.lower() not in _SUITE_NOISE:
            if word != LIVE.current:
                LIVE.progress = (0, 0)
            LIVE.current = word


def _collect(work: Path, spec: RunSpec, settings: Settings, cancelled: bool = False) -> Path:
    """Moves gbench's run directory to the results directory as <run_id>/ and writes gbtui.json."""
    dest = settings.results_path / LIVE.run_id
    dest.mkdir(parents=True, exist_ok=True)
    stamped = [d for d in work.iterdir() if d.is_dir()]
    for d in stamped:
        for item in d.iterdir():
            target = dest / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(d, ignore_errors=True)
    for f in work.iterdir():
        if f.is_file():
            shutil.move(str(f), str(dest / f.name))
    shutil.rmtree(work, ignore_errors=True)
    spec_d = asdict(spec)
    spec_d["endpoint"] = {k: v for k, v in spec_d["endpoint"].items() if k not in ("api_key_env",)}
    if not spec_d["endpoint"].get("quant"):
        spec_d["endpoint"]["quant"] = spec.endpoint.effective_quant
    meta = {"run_id": LIVE.run_id, "spec": spec_d, "command": gbench_args(spec, "<results>"),
            "started": LIVE.started, "finished": time.time(), "rc": LIVE.rc,
            "status": "cancelled" if cancelled else ("ok" if LIVE.rc in (0, 1) else "failed"),
            "error": LIVE.error, "sandbox": {"memory_gb": settings.sandbox_memory_gb, "cpus": settings.cpus(),
                                            "template": sandbox.TEMPLATE},
            "gbench_commit": sandbox.gbench_commit()}
    (dest / "gbtui.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest
