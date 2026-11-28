"""Harbor runner for real agentic harness evaluation (Pi, OpenCode, Hermes, …) on Terminal-Bench / SWE-bench.

Harbor orchestrates per-task Docker containers on the host Docker daemon and connects the unmodified coding agent
inside each task container back to our local `llama-server` (or any OpenAI-compatible `/v1` endpoint) via
`--base-url http://host.docker.internal:<port>/v1 --api-format openai_chat_completions`.

After a Harbor job finishes, `parse_job()` reads Harbor's `result.json`, trial `result.json` files and ATIF
`trajectory.json` logs, emitting a normalized `summary.json` (`benchmark_type: "harness"`) and `harbor_trials.json`
so Results, per-sample inspection, Leaderboard (including the Adapter × Harness heatmap matrix) and Dataset Export
handle Harbor runs natively.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from . import sandbox
from .config import Endpoint

HARNESSES = [
    ("pi", "Pi", "Minimal, token-efficient coding agent (earendil-works/pi)"),
    ("opencode", "OpenCode", "Open-source terminal coding agent with snapshotting"),
    ("hermes", "Hermes", "Nous Research general assistant agent harness"),
    ("claude-code", "Claude Code", "Anthropic coding harness"),
    ("openhands", "OpenHands", "OpenHands software engineering agent"),
    ("aider", "Aider", "Aider code editing harness"),
]

DATASETS = [
    ("terminal-bench@2.0", "Terminal-Bench 2.0", "CLI, systems, data & debugging tasks in Docker"),
    ("swe-bench-verified", "SWE-bench Verified", "Real GitHub issue resolution with hidden unit tests"),
    ("swe-bench-pro", "SWE-bench Pro", "Multi-language repository PR tasks"),
]


def installed() -> bool:
    return shutil.which("harbor") is not None


def version() -> str:
    if not installed():
        return ""
    try:
        r = subprocess.run(["harbor", "--version"], capture_output=True, text=True, timeout=10, env=sandbox._env())
        out = (r.stdout or r.stderr or "").strip().splitlines()
        return out[-1].strip() if out else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def status() -> str | None:
    """None when Harbor runs can start; otherwise a concise setup hint."""
    if not installed():
        return "Harbor is not installed: run `uv tool install harbor`"
    if not sandbox.installed():
        return "Docker is not installed (Harbor requires Docker)"
    try:
        r = sandbox.docker("info", "--format", "{{.ServerVersion}}", timeout=15)
        if r.returncode != 0:
            return "Docker daemon not running (start Docker Desktop / OrbStack / Colima)"
    except (OSError, subprocess.SubprocessError) as e:
        return f"Docker not answering: {e}"
    return None


def harbor_url(base_url: str) -> str:
    """Endpoint URL as seen from inside Harbor's Docker task containers."""
    return sandbox.sandbox_url(base_url)


def model_arg(ep: Endpoint, harness: str = "") -> str:
    """Formats endpoint model id for Harbor.

    OpenCode's built-in `openai` provider routes through the OpenAI Responses API (`t.responses()`), which fails when
    Harbor configures `@ai-sdk/openai-compatible` for a custom `--base-url`. Using a non-builtin provider prefix
    (`local/<model>`) makes OpenCode call `/v1/chat/completions` cleanly, while Pi / Hermes use `openai/<model>`.
    """
    raw = (ep.model or ep.name or "default").strip()
    agent = str(harness or "").strip().lower()
    if "/" in raw:
        prov = raw.split("/", 1)[0].lower()
        if prov in ("openai", "anthropic", "google", "gemini", "openrouter", "bedrock", "amazon-bedrock", "local"):
            return raw
    if agent in ("opencode", "opencode-v2"):
        return f"local/{raw}"
    return f"openai/{raw}"


def dataset_short(dataset: str) -> str:
    """Compact display tag for a dataset string or local path."""
    s = str(dataset or "").strip().rstrip("/")
    if not s:
        return "harbor"
    if "/" in s and Path(s).expanduser().exists():
        return Path(s).expanduser().name
    return s.split("@", 1)[0].rsplit("/", 1)[-1]


CODE_MODE_TOOLS = frozenset({
    "bash", "shell", "terminal", "execute_bash", "execute_code", "run_code",
    "code_exec", "codemode", "execute_ipython_cell", "python", "eval_js", "script", "run_shell",
})

CODE_MODE_INSTRUCTION = (
    "Prefer Code Mode (programmatic tool orchestration): batch multi-step file inspection, filtering, "
    "computation and verification into concise shell/python scripts in single execution turns rather than "
    "issuing many separate read/grep/edit tool calls."
)


def classify_tool_call(name: str) -> str:
    """Returns 'code_mode' for programmatic/execution tools, else 'normal'."""
    n = str(name or "").strip().lower().split(".")[-1].split(":")[-1]
    if n in CODE_MODE_TOOLS or any(k in n for k in ("codemode", "code_exec", "exec_code", "ipython", "bash", "terminal")):
        return "code_mode"
    return "normal"


def build_args(
    ep: Endpoint,
    harness: str,
    dataset: str,
    jobs_dir: Path,
    job_name: str,
    limit: int = 0,
    tasks: list[str] | None = None,
    concurrency: int = 1,
    max_output_tokens: int = 4096,
    tool_mode: str = "normal",
) -> list[str]:
    """Builds the exact `harbor run` CLI command for an endpoint + harness + dataset."""
    cmd = ["harbor", "run"]
    ds = str(dataset or "terminal-bench@2.0").strip()
    ds_path = Path(ds).expanduser()
    if ds_path.exists():
        cmd += ["--path", str(ds_path)]
    else:
        cmd += ["--dataset", ds]
    agent = str(harness or "pi").strip().lower()
    cmd += [
        "--agent", agent,
        "--model", model_arg(ep, agent),
        "--base-url", harbor_url(ep.base_url),
        "--api-format", "openai_chat_completions",
        "--env", "docker",
        "--n-concurrent", str(max(1, int(concurrency or 1))),
        "--jobs-dir", str(jobs_dir),
        "--job-name", job_name,
        "--yes",
        "--delete",
    ]
    # On Apple Silicon / arm64, prebuilt terminal-bench x64 images crash native x64 installers (opencode npm, hermes uv,
    # claude-code) under Rosetta 2: building the task Dockerfile natively for linux/arm64 avoids emulation cleanly.
    if agent in ("opencode", "opencode-v2", "hermes", "claude-code") and platform.machine().lower() in ("arm64", "aarch64"):
        cmd.append("--force-build")
    if agent == "pi":
        if ep.ctx and ep.ctx > 0:
            cmd += ["--ak", f"context_window={ep.ctx}"]
        if max_output_tokens and max_output_tokens > 0:
            cmd += ["--ak", f"max_output_tokens={max_output_tokens}"]
    if str(tool_mode or "normal").lower() == "code_mode":
        if agent == "hermes":
            cmd += ["--ak", "toolsets=codemode,terminal,file"]
        cmd += ["--extra-instruction", CODE_MODE_INSTRUCTION]
    for pat in (tasks or []):
        pat = str(pat).strip()
        if pat:
            cmd += ["--include-task-name", pat]
    if limit and limit > 0:
        cmd += ["--n-tasks", str(int(limit))]
    return cmd


def build_env(ep: Endpoint) -> dict[str, str]:
    """Environment for `harbor run`: strips unrelated host secrets, injects endpoint API key (or 'local')."""
    env = sandbox._env()
    key = os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""
    env["OPENAI_API_KEY"] = key or "local"
    env["PYTHONUNBUFFERED"] = "1"
    env["COLUMNS"] = "120"
    return env


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _extract_reward(trial: dict, trial_dir: Path) -> float | None:
    vr = trial.get("verifier_result")
    if isinstance(vr, dict):
        rewards = vr.get("rewards")
        if isinstance(rewards, dict) and rewards:
            if isinstance(rewards.get("reward"), (int, float)):
                return float(rewards["reward"])
            for v in rewards.values():
                if isinstance(v, (int, float)):
                    return float(v)
    # Fallback: read verifier/reward.json or verifier/reward.txt directly
    rj = _load_json(trial_dir / "verifier" / "reward.json")
    if isinstance(rj, dict) and isinstance(rj.get("reward"), (int, float)):
        return float(rj["reward"])
    rt = trial_dir / "verifier" / "reward.txt"
    if rt.is_file():
        try:
            return float(rt.read_text(encoding="utf-8").strip().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            pass
    return None


def _parse_trajectory(trial_dir: Path) -> dict:
    """Extracts tokens, cache tokens, code-mode vs normal tool calls, tool errors, prompt and response from ATIF."""
    out = {
        "input_tokens": None, "output_tokens": None, "cache_tokens": None,
        "tool_calls": 0, "code_mode_calls": 0, "normal_tool_calls": 0, "tool_errors": 0,
        "tool_counts": {}, "tool_list": [], "prompt": "", "response": "", "steps": 0,
    }
    traj_path = next(iter(sorted(trial_dir.glob("**/trajectory.json"))), None)
    if not traj_path:
        return out
    traj = _load_json(traj_path)
    if not isinstance(traj, dict):
        return out
    steps = traj.get("steps") if isinstance(traj.get("steps"), list) else []
    out["steps"] = len(steps)
    tools: list[dict] = []
    tool_counts: dict[str, int] = {}
    code_calls = normal_calls = tool_errs = 0
    last_agent_msg = ""
    first_user_msg = ""
    sum_in = sum_out = sum_cache = 0
    saw_in = saw_out = saw_cache = False
    for st in steps:
        if not isinstance(st, dict):
            continue
        src = str(st.get("source") or "").lower()
        msg = st.get("message")
        msg_str = msg if isinstance(msg, str) else (json.dumps(msg, ensure_ascii=False) if msg else "")
        if src == "user" and not first_user_msg and msg_str:
            first_user_msg = msg_str
        if src == "agent" and msg_str:
            last_agent_msg = msg_str
        tcs = st.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict):
                    tname = str(tc.get("function_name") or tc.get("name") or "tool")
                    tools.append({"name": tname, "arguments": tc.get("arguments")})
                else:
                    tname = str(tc)
                    tools.append({"name": tname})
                tool_counts[tname] = tool_counts.get(tname, 0) + 1
                if classify_tool_call(tname) == "code_mode":
                    code_calls += 1
                else:
                    normal_calls += 1
        obs = st.get("observation")
        if isinstance(obs, dict):
            for res in (obs.get("results") or []):
                if isinstance(res, dict):
                    txt = str(res.get("content") or "").lower()
                    if res.get("is_error") or "traceback (most recent" in txt or "exit code: 1" in txt or "command failed" in txt:
                        tool_errs += 1
        m = st.get("metrics")
        if isinstance(m, dict):
            if isinstance(m.get("prompt_tokens"), (int, float)):
                sum_in += int(m["prompt_tokens"])
                saw_in = True
            if isinstance(m.get("completion_tokens"), (int, float)):
                sum_out += int(m["completion_tokens"])
                saw_out = True
            if isinstance(m.get("cached_tokens"), (int, float)):
                sum_cache += int(m["cached_tokens"])
                saw_cache = True
    fm = traj.get("final_metrics")
    if isinstance(fm, dict):
        if isinstance(fm.get("total_prompt_tokens"), (int, float)):
            out["input_tokens"] = int(fm["total_prompt_tokens"])
        elif saw_in:
            out["input_tokens"] = sum_in
        if isinstance(fm.get("total_completion_tokens"), (int, float)):
            out["output_tokens"] = int(fm["total_completion_tokens"])
        elif saw_out:
            out["output_tokens"] = sum_out
        if isinstance(fm.get("total_cached_tokens"), (int, float)):
            out["cache_tokens"] = int(fm["total_cached_tokens"])
        elif saw_cache:
            out["cache_tokens"] = sum_cache
    else:
        if saw_in:
            out["input_tokens"] = sum_in
        if saw_out:
            out["output_tokens"] = sum_out
        if saw_cache:
            out["cache_tokens"] = sum_cache
    out["tool_calls"] = len(tools)
    out["code_mode_calls"] = code_calls
    out["normal_tool_calls"] = normal_calls
    out["tool_errors"] = tool_errs
    out["tool_counts"] = tool_counts
    out["tool_list"] = tools
    out["prompt"] = first_user_msg
    out["response"] = last_agent_msg
    return out


def _timing_sec(block) -> float | None:
    if not isinstance(block, dict):
        return None
    a, b = block.get("started_at"), block.get("finished_at")
    if not (isinstance(a, str) and isinstance(b, str)):
        return None
    try:
        ta = datetime.fromisoformat(a.replace("Z", "+00:00"))
        tb = datetime.fromisoformat(b.replace("Z", "+00:00"))
        return max(0.0, (tb - ta).total_seconds())
    except ValueError:
        return None


def parse_job(job_dir: Path, harness: str, dataset: str, tool_mode: str = "normal") -> tuple[dict, list[dict]]:
    """Parses a finished Harbor job directory into a normalized (summary_dict, trials_list)."""
    trials: list[dict] = []
    agg_tools: dict[str, int] = {}
    # Harbor writes <job_dir>/<trial_name>/result.json for each trial
    trial_files = sorted(p for p in job_dir.glob("*/result.json") if p.is_file())
    for tf in trial_files:
        td = tf.parent
        data = _load_json(tf)
        if not isinstance(data, dict):
            continue
        task_name = str(data.get("task_name") or td.name)
        reward = _extract_reward(data, td)
        exc = data.get("exception_info")
        err_msg = ""
        if isinstance(exc, dict):
            et = exc.get("exception_type") or "Error"
            em = str(exc.get("exception_message") or "").strip()
            if len(em) > 380:
                em = em[:100] + " … " + em[-260:]
            err_msg = f"{et}: {em}".strip(": ")
        ar = data.get("agent_result") if isinstance(data.get("agent_result"), dict) else {}
        traj = _parse_trajectory(td)
        in_tok = ar.get("n_input_tokens") if isinstance(ar.get("n_input_tokens"), (int, float)) else traj["input_tokens"]
        out_tok = ar.get("n_output_tokens") if isinstance(ar.get("n_output_tokens"), (int, float)) else traj["output_tokens"]
        cache_tok = ar.get("n_cache_tokens") if isinstance(ar.get("n_cache_tokens"), (int, float)) else traj["cache_tokens"]
        in_tok = int(in_tok) if in_tok is not None else None
        out_tok = int(out_tok) if out_tok is not None else None
        cache_tok = int(cache_tok) if cache_tok is not None else None
        tot_tok = (in_tok or 0) + (out_tok or 0) if (in_tok is not None or out_tok is not None) else None
        tool_calls = traj["tool_calls"]
        code_calls = traj["code_mode_calls"]
        normal_calls = traj["normal_tool_calls"]
        for k, v in (traj.get("tool_counts") or {}).items():
            agg_tools[k] = agg_tools.get(k, 0) + v
        meta = ar.get("metadata") if isinstance(ar.get("metadata"), dict) else {}
        if tool_calls == 0 and isinstance(meta.get("tool_calls"), int):
            tool_calls = meta["tool_calls"]
        resolved = bool(reward is not None and reward >= 1.0 - 1e-9)
        status_str = "passed" if resolved else ("error" if err_msg and reward is None else "failed")
        trials.append({
            "task_name": task_name,
            "trial_name": str(data.get("trial_name") or td.name),
            "reward": reward,
            "correct": resolved,
            "status": status_str,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_tokens": cache_tok,
            "total_tokens": tot_tok,
            "tool_calls": tool_calls,
            "code_mode_calls": code_calls,
            "normal_tool_calls": normal_calls,
            "tool_errors": traj["tool_errors"],
            "tool_call_list": traj["tool_list"],
            "steps": traj["steps"],
            "prompt": traj["prompt"],
            "response": traj["response"],
            "error": err_msg,
            "duration_s": _timing_sec(data.get("agent_execution")),
        })

    # Fallback: if no per-trial result.json was found, inspect top-level job result.json
    job_res = _load_json(job_dir / "result.json")
    n_trials = len(trials)
    n_resolved = sum(1 for t in trials if t["correct"])
    n_errors = sum(1 for t in trials if t["status"] == "error")
    if n_trials == 0 and isinstance(job_res, dict):
        stats = job_res.get("stats") if isinstance(job_res.get("stats"), dict) else {}
        n_trials = int(stats.get("n_completed_trials") or 0)
        n_errors = int(stats.get("n_errored_trials") or 0)
    resolved_pct = (100.0 * n_resolved / n_trials) if n_trials > 0 else None

    tok_vals = [t["total_tokens"] for t in trials if t["total_tokens"] is not None]
    in_vals = [t["input_tokens"] for t in trials if t["input_tokens"] is not None]
    out_vals = [t["output_tokens"] for t in trials if t["output_tokens"] is not None]
    cache_vals = [t["cache_tokens"] for t in trials if t["cache_tokens"] is not None]
    tc_vals = [t["tool_calls"] for t in trials if t["tool_calls"] is not None]
    cm_vals = [t["code_mode_calls"] for t in trials]
    nm_vals = [t["normal_tool_calls"] for t in trials]
    err_tc = sum(t["tool_errors"] for t in trials)
    tot_tc = sum(tc_vals)
    step_vals = [t["steps"] for t in trials if t["steps"] > 0]
    dur_vals = [t["duration_s"] for t in trials if t["duration_s"] is not None]

    avg_tokens = round(sum(tok_vals) / len(tok_vals), 1) if tok_vals else None
    avg_in = round(sum(in_vals) / len(in_vals), 1) if in_vals else None
    avg_out = round(sum(out_vals) / len(out_vals), 1) if out_vals else None
    avg_cache = round(sum(cache_vals) / len(cache_vals), 1) if cache_vals else None
    avg_tools = round(sum(tc_vals) / len(tc_vals), 2) if tc_vals else None
    avg_cm = round(sum(cm_vals) / len(cm_vals), 2) if cm_vals else None
    avg_nm = round(sum(nm_vals) / len(nm_vals), 2) if nm_vals else None
    cm_ratio = round(100.0 * sum(cm_vals) / tot_tc, 1) if tot_tc > 0 else None
    cache_hit = round(100.0 * sum(cache_vals) / sum(in_vals), 1) if (cache_vals and sum(in_vals) > 0) else None
    avg_steps = round(sum(step_vals) / len(step_vals), 1) if step_vals else None
    tok_per_turn = round(sum(tok_vals) / sum(step_vals), 1) if (tok_vals and step_vals and sum(step_vals) > 0) else None
    tok_per_resolved = round(sum(tok_vals) / n_resolved, 1) if (tok_vals and n_resolved > 0) else None
    tool_err_rate = round(100.0 * err_tc / tot_tc, 1) if tot_tc > 0 else None
    avg_dur = round(sum(dur_vals) / len(dur_vals), 1) if dur_vals else None

    ds_short = dataset_short(dataset)
    entry = {
        "benchmark_type": "harness",
        "harness": str(harness or "pi").lower(),
        "tool_mode": str(tool_mode or "normal").lower(),
        "dataset": dataset,
        "dataset_short": ds_short,
        "resolved_rate": round(resolved_pct, 2) if resolved_pct is not None else None,
        "resolved_tasks": n_resolved,
        "total_tasks": n_trials,
        "error_tasks": n_errors,
        "avg_tokens": avg_tokens,
        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,
        "avg_cache_tokens": avg_cache,
        "cache_hit_rate_pct": cache_hit,
        "avg_tool_calls": avg_tools,
        "avg_code_mode_calls": avg_cm,
        "avg_normal_tool_calls": avg_nm,
        "code_mode_ratio_pct": cm_ratio,
        "tool_error_rate_pct": tool_err_rate,
        "tool_distribution": dict(sorted(agg_tools.items(), key=lambda kv: -kv[1])[:12]),
        "avg_steps": avg_steps,
        "tokens_per_turn": tok_per_turn,
        "tokens_per_resolved": tok_per_resolved,
        "avg_duration_s": avg_dur,
        "status": "completed" if (n_trials > 0 and n_errors < n_trials) else "failed",
        "trials_file": "harbor_trials.json",
    }
    return {"models": [entry], "failures": []}, trials


def parse_progress_line(line: str) -> tuple[int, int, str] | None:
    """Extracts (done, total, current_task) from Harbor's stdout lines when present."""
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+(?:trials?|tasks?)", line, re.I)
    if m:
        return int(m.group(1)), int(m.group(2)), ""
    m2 = re.search(r"(?:Running|Starting|Finished)\s+trial\s+['\"]?([A-Za-z0-9_.-]+)['\"]?", line, re.I)
    if m2:
        return 0, 0, m2.group(1)[:40]
    return None
