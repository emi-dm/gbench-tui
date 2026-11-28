"""A run: either gbench's CLI inside a fresh hardened Docker container, or Harbor for real agentic harness evaluation.

Results: every run is collected into `<results_dir>/<run_id>/` with `gbtui.json` + `summary.json` (+ gbench's own
`eval_*.json` / `metadata.json`, or Harbor's `harbor_trials.json` / `harbor_result.json`), so Results, Leaderboard,
the Adapter × Harness heatmap matrix and Dataset Export treat both uniformly.
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

from . import harbor_runner, notify, sandbox, servers, sysmem
from .config import Endpoint, Settings, config_dir

TIERS = {
    "golden": "Golden Set: 16 deterministic capability smoke tests",
    "evals": "Academic eval suites (pick suites; stratified subset with --eval-limit)",
    "perf": "Performance on this device (--personal: single-stream decode tok/s and TTFT)",
    "preflight": "Preflight: every selected suite's prerequisites, no model request",
    "harness": "Agent harness benchmark via Harbor (Pi, OpenCode, Hermes on Terminal-Bench / SWE-bench)",
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
    harness: str = "pi"
    harness_dataset: str = "terminal-bench@2.0"
    harness_tasks: list = field(default_factory=list)
    tool_mode: str = "normal"       # normal | code_mode
    stop_server_after: bool = False  # stop llama-server after this spec (used by Quant Sweeps)

    def run_id(self) -> str:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = self.endpoint.name
        q = self.endpoint.effective_quant
        if q and q.lower() not in tag.lower().replace("-", "_"):
            tag = f"{tag}-{q}"
        ad = self.endpoint.effective_adapter
        if ad and ad.lower() not in tag.lower().replace("_", "-"):
            tag = f"{tag}-{ad}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)[:46]
        if self.tier == "harness":
            h_slug = re.sub(r"[^a-z0-9-]+", "-", (self.harness or "pi").lower()).strip("-")
            mode_sfx = "-code" if self.tool_mode == "code_mode" else ""
            return f"{ts}-{safe}-harness-{h_slug}{mode_sfx}"
        return f"{ts}-{safe}-{self.tier}"


def gbench_args(spec: RunSpec, results_dir: str) -> list[str]:
    """The gbench command line for a spec (the URL as the container sees it)."""
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
    if spec.tier == "harness":
        args = harbor_runner.build_args(
            spec.endpoint, spec.harness, spec.harness_dataset,
            Path("<jobs>"), "<run_id>",
            limit=spec.eval_limit, tasks=spec.harness_tasks,
            concurrency=spec.concurrency, max_output_tokens=spec.max_output_tokens,
            tool_mode=spec.tool_mode,
        )
        return " ".join(args)
    return " ".join(gbench_args(spec, f"{sandbox.CONTAINER_WORK}/runs/<id>"))


def specs_for_preset(ep: Endpoint, preset_name: str, settings: Settings, stop_after: bool = False) -> list[RunSpec]:
    """Expands a preset name (e.g. 'smoke', 'quant-shootout') into an ordered list of RunSpecs for `ep`."""
    preset = settings.all_presets().get(preset_name)
    if not preset:
        raise KeyError(f"unknown preset {preset_name!r}")
    d = settings.run
    out: list[RunSpec] = []
    steps = preset.get("steps") or []
    for i, st in enumerate(steps):
        tier = st.get("tier", "golden")
        out.append(RunSpec(
            endpoint=ep,
            tier=tier,
            evals=list(st.get("evals") or d.evals),
            golden_tasks=list(st.get("golden_tasks") or []),
            eval_limit=int(st.get("eval_limit", d.harness_limit if tier == "harness" else d.eval_limit)),
            max_output_tokens=int(st.get("max_output_tokens", d.max_output_tokens)),
            thinking=bool(st.get("thinking", d.thinking)),
            temperature=str(st.get("temperature", d.temperature)),
            concurrency=int(st.get("concurrency", 1 if tier == "harness" else d.concurrency)),
            suite_timeout=int(st.get("suite_timeout", d.suite_timeout)),
            perf_mode=str(st.get("perf_mode", d.perf_mode)),
            harness=str(st.get("harness", d.harness)),
            harness_dataset=str(st.get("harness_dataset", d.harness_dataset)),
            harness_tasks=list(st.get("harness_tasks") or []),
            tool_mode=str(st.get("tool_mode", d.tool_mode)),
            stop_server_after=(stop_after and i == len(steps) - 1),
        ))
    return out


def build_sweep_specs(endpoints: list[Endpoint], preset_name: str, settings: Settings, stop_between: bool = True) -> list[RunSpec]:
    """Builds a sequential Quantization / Endpoint Sweep: start server -> run preset tiers -> stop server -> next."""
    specs: list[RunSpec] = []
    for ep in endpoints:
        specs.extend(specs_for_preset(ep, preset_name, settings, stop_after=stop_between))
    return specs


def estimate_duration_seconds(spec: RunSpec, runs: list) -> tuple[float | None, str]:
    """Estimates approximate wall-clock duration in seconds using measured `decode tok/s` from prior runs."""
    tok_s: float | None = None
    src = ""
    ep = spec.endpoint
    for r in runs:
        if r.status != "ok":
            continue
        same_ep = (r.model == ep.name) or (r.base_model.lower() == ep.base_model.lower() and r.quant == ep.effective_quant)
        if not same_ep:
            continue
        for s in r.scores:
            if s.kind == "perf" and "decode tok/s" in s.name.lower() and s.value and s.value > 0:
                tok_s = float(s.value)
                src = f"{tok_s:.0f} tok/s"
                break
        if tok_s:
            break
    if not tok_s:
        return None, ""
    if spec.tier == "preflight":
        return 8.0, "preflight (no model calls)"
    if spec.tier == "perf":
        n_req = 12 if spec.perf_mode == "personal" else 30
        return round(n_req * (512 / tok_s) + 12.0, 1), src
    if spec.tier == "golden":
        n_tasks = len(spec.golden_tasks) if spec.golden_tasks else 14
        avg_tok = 450 if spec.thinking else 240
        return round((n_tasks * avg_tok / tok_s) + 12.0, 1), src
    if spec.tier == "evals":
        n_suites = max(1, len(spec.evals or ["gsm8k"]))
        n_per = spec.eval_limit if spec.eval_limit > 0 else 100
        avg_tok = min(spec.max_output_tokens, 900 if spec.thinking else 380)
        conc = max(1, spec.concurrency)
        eff = max(1.0, conc * 0.75)
        return round((n_suites * n_per * avg_tok) / (tok_s * eff) + 15.0 * n_suites, 1), src
    if spec.tier == "harness":
        n_tasks = spec.eval_limit if spec.eval_limit > 0 else 5
        avg_tok = 6500
        conc = max(1, spec.concurrency)
        return round((n_tasks * (avg_tok / tok_s + 75.0)) / conc, 1), src
    return None, ""


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
    progress: tuple = (0, 0)       # (done, total) from the last tqdm / golden / harbor line
    current: str = ""              # suite / task being measured
    sandbox: str = ""
    result_dir: str = ""
    error: str = ""
    proc: subprocess.Popen | None = None
    cancel: bool = False
    _golden_seen: set = field(default_factory=set)
    _thread: threading.Thread | None = None
    matrix_pos: tuple = (0, 0)     # (current_cell, total_cells) during Adapter × Harness matrix runs

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
            "result_dir": LIVE.result_dir, "error": LIVE.error, "matrix_pos": LIVE.matrix_pos,
        }


def wait(timeout: float | None = None) -> None:
    """Wait for the active background run thread (including container cleanup) to finish."""
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
    """Starts a single run in a background thread. None when started, else why not."""
    return start_batch([spec], settings)


def start_batch(specs: list[RunSpec], settings: Settings) -> str | None:
    """Starts one or more sequential runs (e.g. an Adapter × Harness matrix) in a background thread."""
    if not specs:
        return "nothing to run"
    with _LOCK:
        if LIVE.active or (LIVE._thread and LIVE._thread.is_alive()):
            return "a run is already in progress (one at a time, to keep the machine responsive)"
        if other_run():
            return "another gbench-tui is running a benchmark (one at a time)"
        rid = specs[0].run_id()
        started = time.time()
        try:
            lock_file().write_text(json.dumps({"pid": os.getpid(), "run_id": rid, "started": started}), encoding="utf-8")
        except OSError as e:
            return f"could not acquire run lock: {e}"
        LIVE.__init__()
        LIVE.spec, LIVE.run_id, LIVE.phase, LIVE.started = specs[0], rid, "server", started
        LIVE.matrix_pos = (1, len(specs)) if len(specs) > 1 else (0, 0)
        t = threading.Thread(target=_run_batch, args=(specs, settings), daemon=True)
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


def _ensure_endpoint(ep: Endpoint, settings: Settings) -> None:
    if ep.gguf:
        # Ensure the exact GGUF + LoRA for this endpoint is serving (restarts if port has another variant)
        if not (servers.server_pid(ep) and servers._config_matches_running(ep) and servers.probe(ep, timeout=1.5)["ok"]):
            _say(f"▸ starting llama-server for {ep.name}…")
            why = servers.start_server(ep, settings.min_free_gb, on_line=_say)
            if why:
                raise RuntimeError(why)
    pr = servers.probe(ep)
    if not pr["ok"]:
        raise RuntimeError(f"endpoint {ep.base_url} not reachable: {pr['error']}")
    _say(f"✔ endpoint {ep.base_url} serves: {', '.join(pr['models'][:4])}")


def _stream_proc(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=env, bufsize=0)
        LIVE.proc = p
        buf = b""
        assert p.stdout is not None
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
        return p.wait()


def _run_batch(specs: list[RunSpec], settings: Settings) -> None:
    last_phase = "failed"
    try:
        for idx, spec in enumerate(specs, 1):
            if LIVE.cancel:
                last_phase = "cancelled"
                break
            rid = spec.run_id()
            with _LOCK:
                LIVE.spec = spec
                LIVE.run_id = rid
                LIVE.phase = "server"
                LIVE.rc = None
                LIVE.error = ""
                LIVE.progress = (0, 0)
                LIVE.current = ""
                LIVE.sandbox = ""
                if len(specs) > 1:
                    LIVE.matrix_pos = (idx, len(specs))
            if len(specs) > 1:
                ad = spec.endpoint.effective_adapter or "base"
                _say(f"━━━ [{idx}/{len(specs)}] {spec.endpoint.name} ({ad}) × {spec.harness or spec.tier} ━━━")
            last_phase = _run_one(spec, settings, rid)
            if spec.stop_server_after and spec.endpoint.gguf:
                _say(f"■ stopping llama-server for {spec.endpoint.name} (sweep step complete)")
                servers.stop_server(spec.endpoint)
            if LIVE.cancel:
                last_phase = "cancelled"
                break
            if idx < len(specs):
                time.sleep(1.0)                     # ensure distinct timestamped run_id
    finally:
        lock_file().unlink(missing_ok=True)
        with _LOCK:
            LIVE.finished = time.time()
            LIVE.proc = None
            LIVE.phase = last_phase
        if getattr(settings, "notify_on_finish", True):
            title = "gbench-tui · sweep finished" if len(specs) > 1 else "gbench-tui · run finished"
            body = f"{specs[-1].endpoint.name} ({last_phase}, {len(specs)} run(s))"
            notify.notify(title, body)


def _run_one(spec: RunSpec, settings: Settings, rid: str) -> str:
    if spec.tier == "harness":
        return _run_harbor(spec, settings, rid)
    return _run_gbench(spec, settings, rid)


def _run_gbench(spec: RunSpec, settings: Settings, rid: str) -> str:
    ep = spec.endpoint
    box = ""
    work: Path | None = None
    final_phase = "failed"
    try:
        work = sandbox.work_dir() / "runs" / rid
        work.mkdir(parents=True, exist_ok=True)
        _ensure_endpoint(ep, settings)
        if LIVE.cancel:
            raise InterruptedError
        LIVE.phase = "sandbox"
        why = sysmem.check(settings.sandbox_memory_gb * sysmem.GB, settings.min_free_gb)
        if why:
            raise RuntimeError(why)
        box = sandbox.create(settings.sandbox_memory_gb, settings.cpus(), sandbox.allow_for(ep.base_url), _say)
        LIVE.sandbox = box
        if LIVE.cancel:
            raise InterruptedError
        LIVE.phase = "running"
        cwdir = sandbox.CONTAINER_WORK
        args = gbench_args(spec, f"{cwdir}/runs/{rid}")
        _say("$ " + " ".join(args))
        env = {"HF_HOME": f"{cwdir}/hf", "NLTK_DATA": "/opt/nltk_data", "PYTHONUNBUFFERED": "1",
               "TQDM_MININTERVAL": "2", "TOKENIZERS_PARALLELISM": "false", "COLUMNS": "120"}
        key = os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""
        if key:
            env["VLLM_API_KEY"] = key
        cmd = sandbox.exec_argv(box, args, env=env)
        LIVE.rc = _stream_proc(cmd, sandbox._env(), work / "gbench.log")
        if LIVE.cancel:
            raise InterruptedError
        LIVE.phase = "collecting"
        LIVE.result_dir = str(_collect(work, spec, settings, rid))
        final_phase = "done" if LIVE.rc in (0, 1) else "failed"
        verdict = {0: "every pillar produced its result", 1: "the model failed some Golden Set cases",
                   2: "a harness error (see the log): not a verdict on the model"}.get(LIVE.rc, f"exit code {LIVE.rc}")
        _say(f"{'✔' if LIVE.rc in (0, 1) else '✗'} gbench finished: {verdict}")
    except InterruptedError:
        final_phase = "cancelled"
        LIVE.phase = "collecting"
        _say("■ cancelled")
        if work and work.is_dir() and any(work.iterdir()):
            LIVE.result_dir = str(_collect(work, spec, settings, rid, cancelled=True))
    except Exception as e:                           # noqa: BLE001 - shown in the live view
        final_phase = "failed"
        LIVE.error = str(e)
        LIVE.phase = "collecting"
        _say(f"✗ {e}")
        if work and work.is_dir() and any(work.iterdir()):
            LIVE.result_dir = str(_collect(work, spec, settings, rid))
    finally:
        if box:
            _say(f"removing container {box}…")
            sandbox.destroy(box)
        try:
            from . import leaderboard
            leaderboard.write(settings)
        except Exception:                             # noqa: BLE001
            pass
    return final_phase


def _run_harbor(spec: RunSpec, settings: Settings, rid: str) -> str:
    ep = spec.endpoint
    jobs_root = sandbox.work_dir() / "harbor-jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    job_dir = jobs_root / rid
    work = sandbox.work_dir() / "runs" / rid
    work.mkdir(parents=True, exist_ok=True)
    final_phase = "failed"
    try:
        st = harbor_runner.status()
        if st:
            raise RuntimeError(st)
        _ensure_endpoint(ep, settings)
        if LIVE.cancel:
            raise InterruptedError
        LIVE.phase = "sandbox"
        why = sysmem.check(settings.sandbox_memory_gb * sysmem.GB, settings.min_free_gb)
        if why:
            raise RuntimeError(why)
        LIVE.phase = "running"
        LIVE.current = f"{spec.harness} · {harbor_runner.dataset_short(spec.harness_dataset)}"
        cmd = harbor_runner.build_args(
            ep, spec.harness, spec.harness_dataset,
            jobs_root, rid,
            limit=spec.eval_limit, tasks=spec.harness_tasks,
            concurrency=spec.concurrency, max_output_tokens=spec.max_output_tokens,
            tool_mode=spec.tool_mode,
        )
        _say("$ " + " ".join(cmd))
        LIVE.rc = _stream_proc(cmd, harbor_runner.build_env(ep), work / "gbench.log")
        if LIVE.cancel:
            raise InterruptedError
        LIVE.phase = "collecting"
        summary, trials = harbor_runner.parse_job(job_dir, spec.harness, spec.harness_dataset, tool_mode=spec.tool_mode)
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (work / "harbor_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
        if (job_dir / "result.json").is_file():
            shutil.copy2(job_dir / "result.json", work / "harbor_result.json")
        # Preserve ATIF trajectories for multi-turn knowledge distillation & live system-prompt inspection
        traj_out = work / "trajectories"
        for td in sorted(job_dir.iterdir()) if job_dir.is_dir() else []:
            tf = td / "agent" / "trajectory.json"
            if td.is_dir() and tf.is_file():
                traj_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tf, traj_out / f"{td.name}.json")
        ent = summary["models"][0]
        ok_trials = ent.get("total_tasks", 0) > 0 and ent.get("status") == "completed"
        LIVE.result_dir = str(_collect(work, spec, settings, rid, command=cmd, backend="harbor", force_failed=not ok_trials))
        final_phase = "done" if (LIVE.rc == 0 and ok_trials) else "failed"
        res_s = f"{ent['resolved_tasks']}/{ent['total_tasks']} resolved ({ent['resolved_rate']:.1f}%)" if ent.get("resolved_rate") is not None else "no trials"
        tok_s = f" · avg {ent['avg_tokens']:.0f} tok" if ent.get("avg_tokens") is not None else ""
        tc_s = f" · avg {ent['avg_tool_calls']:.1f} tools" if ent.get("avg_tool_calls") is not None else ""
        _say(f"{'✔' if final_phase == 'done' else '✗'} harbor ({spec.harness}) finished: {res_s}{tok_s}{tc_s}")
    except InterruptedError:
        final_phase = "cancelled"
        LIVE.phase = "collecting"
        _say("■ cancelled")
        if job_dir.is_dir():
            summary, trials = harbor_runner.parse_job(job_dir, spec.harness, spec.harness_dataset)
            (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (work / "harbor_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
        if work.is_dir() and any(work.iterdir()):
            LIVE.result_dir = str(_collect(work, spec, settings, rid, cancelled=True, backend="harbor"))
    except Exception as e:                           # noqa: BLE001
        final_phase = "failed"
        LIVE.error = str(e)
        LIVE.phase = "collecting"
        _say(f"✗ {e}")
        if work.is_dir() and any(work.iterdir()):
            LIVE.result_dir = str(_collect(work, spec, settings, rid, backend="harbor"))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
        try:
            from . import leaderboard
            leaderboard.write(settings)
        except Exception:                             # noqa: BLE001
            pass
    return final_phase


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
    hp = harbor_runner.parse_progress_line(line)
    if hp:
        d, t, cur = hp
        if t > 0:
            LIVE.progress = (min(d, t), t)
        if cur:
            LIVE.current = cur
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


def _collect(work: Path, spec: RunSpec, settings: Settings, rid: str, cancelled: bool = False,
             command: list[str] | None = None, backend: str = "gbench", force_failed: bool = False) -> Path:
    """Moves run outputs to the results directory as <run_id>/ and writes gbtui.json."""
    dest = settings.results_path / rid
    dest.mkdir(parents=True, exist_ok=True)
    stamped = [d for d in work.iterdir() if d.is_dir()]
    for d in stamped:
        if d.name == "trajectories":
            target = dest / "trajectories"
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(d), str(target))
            continue
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
    if not spec_d["endpoint"].get("adapter_name"):
        spec_d["endpoint"]["adapter_name"] = spec.endpoint.effective_adapter
    if not spec_d["endpoint"].get("target_harness"):
        spec_d["endpoint"]["target_harness"] = spec.endpoint.effective_harness
    ok_rc = (0,) if backend == "harbor" else (0, 1)
    status_val = "cancelled" if cancelled else ("failed" if force_failed else ("ok" if LIVE.rc in ok_rc else "failed"))
    meta = {
        "run_id": rid, "backend": backend, "spec": spec_d,
        "command": command or gbench_args(spec, "<results>"),
        "started": LIVE.started, "finished": time.time(), "rc": LIVE.rc,
        "status": status_val,
        "error": LIVE.error,
        "sandbox": {"memory_gb": settings.sandbox_memory_gb, "cpus": settings.cpus(),
                    "image": sandbox.IMAGE, "template": sandbox.TEMPLATE,
                    "harbor_version": harbor_runner.version() if backend == "harbor" else ""},
        "gbench_commit": sandbox.gbench_commit() if backend == "gbench" else "",
    }
    (dest / "gbtui.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest
