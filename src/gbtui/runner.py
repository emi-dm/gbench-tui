"""A run: gbench's own CLI, inside a fresh Docker Sandbox, against one endpoint.

The command line is plain gbench (`gbench --golden-only|--evals-only|--personal --remote-endpoint … --models …
--tokenizer …`): gbench-tui only chooses the flags, starts the sandbox, streams the output and files the results.

Results: gbench writes to /work/runs/<id>/ (a bind mount of ~/.cache/gbench-tui/work); when the run ends its
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
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.endpoint.name)[:40]
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
        a += ["--personal"] if spec.perf_mode == "personal" else ["--serving-only", "--preset", "quick", "--text-only",
                                                                  "--no-stress-test"]
    if spec.thinking:
        a += ["--eval-thinking"]
    if str(spec.temperature).strip():
        a += ["--temperature", str(spec.temperature).strip()]
    return a


def command_preview(spec: RunSpec) -> str:
    return " ".join(gbench_args(spec, f"{sandbox.WORK}/runs/<id>"))


# ------------------------------------------------------------------------------------------------
# live state of the run in progress (one at a time: protects the machine)
# ------------------------------------------------------------------------------------------------
@dataclass
class Live:
    run_id: str = ""
    spec: RunSpec | None = None
    lines: list = field(default_factory=list)
    phase: str = "idle"            # idle | server | sandbox | running | collecting | done | failed | cancelled
    started: float = 0.0
    finished: float = 0.0
    rc: int | None = None
    progress: tuple = (0, 0)       # (done, total) from the last tqdm line
    current: str = ""              # suite / task being measured
    sandbox: str = ""
    result_dir: str = ""
    error: str = ""
    proc: subprocess.Popen | None = None
    cancel: bool = False

    @property
    def active(self) -> bool:
        return self.phase in ("server", "sandbox", "running", "collecting")

    def add(self, line: str) -> None:
        self.lines.append(line)
        if len(self.lines) > 4000:
            del self.lines[:1000]


LIVE = Live()
_LOCK = threading.Lock()
_TQDM = re.compile(r"(?:(?P<desc>[^\r\n|]{0,60}):\s*)?\s*\d+%\|[^|]*\|\s*(?P<done>\d+)/(?P<total>\d+)")
_SUITE = re.compile(r"(?:Running|Evaluating|Starting)\s+(?:eval(?:uation)?\s+suite\s+)?['\"]?([a-z0-9_]+)['\"]?", re.I)


def lock_file() -> Path:
    return config_dir() / "run.lock"


def other_run() -> dict | None:
    """A run started by another gbench-tui process that is still alive."""
    try:
        st = json.loads(lock_file().read_text())
    except (OSError, ValueError):
        return None
    pid = st.get("pid")
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
        if LIVE.active:
            return "a run is already in progress (one at a time, to keep the machine responsive)"
        if other_run():
            return "another gbench-tui is running a benchmark (one at a time)"
        LIVE.__init__()
        LIVE.spec, LIVE.run_id, LIVE.phase, LIVE.started = spec, spec.run_id(), "server", time.time()
    lock_file().write_text(json.dumps({"pid": os.getpid(), "run_id": LIVE.run_id, "started": LIVE.started}))
    threading.Thread(target=_run, args=(spec, settings), daemon=True).start()
    return None


def cancel() -> None:
    LIVE.cancel = True
    p = LIVE.proc
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGINT)
        except OSError:
            pass
    if LIVE.sandbox:                               # stops gbench and everything it started, at once
        threading.Thread(target=sandbox.destroy, args=(LIVE.sandbox,), daemon=True).start()


def _say(msg: str) -> None:
    LIVE.add(msg)


def _run(spec: RunSpec, settings: Settings) -> None:
    ep = spec.endpoint
    work = sandbox.work_dir() / "runs" / LIVE.run_id
    work.mkdir(parents=True, exist_ok=True)
    box = ""
    try:
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
        args = gbench_args(spec, f"{sandbox.WORK}/runs/{LIVE.run_id}")
        _say("$ " + " ".join(args))
        env = {"HF_HOME": f"{sandbox.WORK}/hf", "NLTK_DATA": "/opt/nltk_data", "PYTHONUNBUFFERED": "1",
               "TQDM_MININTERVAL": "2", "TOKENIZERS_PARALLELISM": "false", "COLUMNS": "120"}
        key = os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""
        if key:
            env["VLLM_API_KEY"] = key
        cmd = sandbox.exec_argv(box, ["sh", "-c", f"cd {sandbox.WORK} && exec " + " ".join(_q(x) for x in args)], env=env)
        log = open(work / "gbench.log", "w")
        LIVE.proc = p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=sandbox._env(), bufsize=0)
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
                if not line:
                    continue
                log.write(line + "\n")
                _parse_progress(line)
                if not _TQDM.search(line):
                    _say(line)
                elif LIVE.lines and _TQDM.search(LIVE.lines[-1]):
                    LIVE.lines[-1] = line
                else:
                    _say(line)
        LIVE.rc = p.wait()
        log.close()
        if LIVE.cancel:
            raise InterruptedError
        LIVE.phase = "collecting"
        LIVE.result_dir = str(_collect(work, spec, settings))
        LIVE.phase = "done" if LIVE.rc in (0, 1) else "failed"
        verdict = {0: "every pillar produced its result", 1: "the model failed some Golden Set cases",
                   2: "a harness error (see the log): not a verdict on the model"}.get(LIVE.rc, f"exit code {LIVE.rc}")
        _say(f"{'✔' if LIVE.rc in (0, 1) else '✗'} gbench finished: {verdict}")
    except InterruptedError:
        LIVE.phase = "cancelled"
        _say("■ cancelled")
        _collect(work, spec, settings, cancelled=True)
    except Exception as e:                           # noqa: BLE001 - shown in the live view
        LIVE.phase, LIVE.error = "failed", str(e)
        _say(f"✗ {e}")
        if any(work.iterdir()):
            LIVE.result_dir = str(_collect(work, spec, settings))
    finally:
        if box:
            _say(f"removing sandbox {box}…")
            sandbox.destroy(box)
        LIVE.finished = time.time()
        LIVE.proc = None
        lock_file().unlink(missing_ok=True)
        try:
            from . import leaderboard
            leaderboard.write(settings)
        except Exception:                             # noqa: BLE001
            pass


def _q(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./:=@,+-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def _parse_progress(line: str) -> None:
    m = _TQDM.search(line)
    if m:
        LIVE.progress = (int(m.group("done")), int(m.group("total")))
        if m.group("desc"):
            LIVE.current = m.group("desc").strip()[-40:]
        return
    m = _SUITE.search(line)
    if m and len(m.group(1)) > 2:
        LIVE.current = m.group(1)


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
    meta = {"run_id": LIVE.run_id, "spec": spec_d, "command": gbench_args(spec, "<results>"),
            "started": LIVE.started, "finished": time.time(), "rc": LIVE.rc,
            "status": "cancelled" if cancelled else ("ok" if LIVE.rc in (0, 1) else "failed"),
            "error": LIVE.error, "sandbox": {"memory_gb": settings.sandbox_memory_gb, "cpus": settings.cpus(),
                                            "template": sandbox.TEMPLATE},
            "gbench_commit": sandbox.gbench_commit()}
    (dest / "gbtui.json").write_text(json.dumps(meta, indent=2))
    return dest
