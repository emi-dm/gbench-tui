"""Reading gbench & Harbor results: one directory per run (gbtui.json + summary.json + eval_*.json / harbor_trials.json).

Everything is normalised into `Run` objects with a flat list of `Score`s (one per Golden Set run, eval suite,
performance measurement, or Harbor harness run) and, on demand, the per-sample items for the failure explorer and
dataset export. Nothing here re-scores anything: the numbers are gbench's or Harbor's.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import catalog, config


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_dict(p: Path) -> dict:
    d = _load(p)
    return d if isinstance(d, dict) else {}


@dataclass
class Score:
    kind: str                  # golden | eval | perf | harness | quality
    name: str                  # "golden_set", suite, "decode tok/s", or "pi:terminal-bench"
    value: float | None        # % for golden / eval / harness, unit's value for perf
    unit: str = "%"
    n: int = 0
    correct: int | None = None
    status: str = ""
    pillar: str = ""
    comparable: bool | None = None
    detail: str = ""
    file: str = ""


@dataclass
class Run:
    id: str
    path: Path
    meta: dict = field(default_factory=dict)          # gbtui.json
    metadata: dict = field(default_factory=dict)      # gbench metadata.json
    summary: dict = field(default_factory=dict)       # gbench / harbor summary.json
    scores: list = field(default_factory=list)

    @property
    def endpoint(self) -> dict:
        return (self.meta.get("spec") or {}).get("endpoint") or {}

    @property
    def model(self) -> str:
        ep = self.endpoint
        if ep.get("name"):
            return ep["name"]
        for r in (self.summary.get("models") or []) + (self.summary.get("failures") or []):
            if not isinstance(r, dict):
                continue
            m = r.get("model_short") or r.get("model") or r.get("model_name")
            if m:
                return str(m)
        return "?"

    @property
    def quant(self) -> str:
        ep = self.endpoint
        if ep.get("quant"):
            q = config.normalize_quant(ep["quant"])
            if q:
                return q
        for cand in (ep.get("gguf"), ep.get("model"), ep.get("name")):
            q = config.detect_quant(cand or "")
            if q:
                return q
        for r in (self.summary.get("models") or []) + (self.summary.get("failures") or []):
            if not isinstance(r, dict):
                continue
            for k in ("model_short", "model", "model_name"):
                q = config.detect_quant(str(r.get(k) or ""))
                if q:
                    return q
        return ""

    @property
    def target_harness(self) -> str:
        ep = self.endpoint
        if ep.get("target_harness"):
            return config.normalize_harness(ep["target_harness"])
        return (config.detect_harness(ep.get("lora") or "")
                or config.detect_harness(ep.get("adapter_name") or ""))

    @property
    def adapter(self) -> str:
        ep = self.endpoint
        if ep.get("adapter_name"):
            return config.normalize_adapter(ep["adapter_name"])
        return config.detect_adapter(
            ep.get("lora") or "", ep.get("model") or "", ep.get("name") or "", self.target_harness,
        )

    @property
    def eval_harness(self) -> str:
        """The agent harness executed during a `tier == 'harness'` Harbor run (empty for standard gbench runs)."""
        if self.tier != "harness":
            return ""
        spec = self.meta.get("spec") or {}
        if spec.get("harness"):
            return config.normalize_harness(spec["harness"])
        for r in (self.summary.get("models") or []):
            if isinstance(r, dict) and r.get("benchmark_type") == "harness" and r.get("harness"):
                return config.normalize_harness(r["harness"])
        return ""

    @property
    def base_model(self) -> str:
        base = config.split_model_quant(self.model, self.quant)[0]
        return config.strip_adapter_suffix(base, self.adapter, self.target_harness)

    @property
    def model_label(self) -> str:
        parts = [p for p in (self.quant, self.adapter) if p]
        if parts:
            return f"{self.base_model} ({' · '.join(parts)})"
        return self.model

    @property
    def tier(self) -> str:
        t = (self.meta.get("spec") or {}).get("tier")
        if t:
            return t
        if any(s.kind == "harness" for s in self.scores):
            return "harness"
        if any(s.kind == "golden" for s in self.scores):
            return "golden"
        if any(s.kind == "eval" for s in self.scores):
            return "evals"
        return "perf"

    @property
    def status(self) -> str:
        return self.meta.get("status") or ("ok" if self.summary else "unknown")

    @property
    def started(self) -> float:
        return self.meta.get("started") or self.path.stat().st_mtime

    @property
    def duration(self) -> float:
        a, b = self.meta.get("started"), self.meta.get("finished")
        return (b - a) if a and b else 0.0

    @property
    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.started))

    @property
    def headline(self) -> str:
        """One line: the main number(s) of the run."""
        parts = []
        for s in self.scores[:4]:
            if s.value is None:
                parts.append(f"{s.name} {s.status or 'n/a'}")
            elif s.unit == "%":
                parts.append(f"{s.name} {s.value:.1f}%")
            elif s.unit == "tok":
                parts.append(f"{s.name} {s.value:.0f} tok")
            else:
                parts.append(f"{s.name} {s.value:.2f} {s.unit}")
        more = len(self.scores) - 4
        return " · ".join(parts) + (f" · +{more}" if more > 0 else "") if parts else (self.meta.get("error") or "no results")

    @property
    def harness_entry(self) -> dict | None:
        """Raw summary dict for a Harbor harness run, if present."""
        for r in (self.summary.get("models") or []):
            if isinstance(r, dict) and r.get("benchmark_type") == "harness":
                return r
        return None

    @property
    def tool_mode(self) -> str:
        he = self.harness_entry
        if he and he.get("tool_mode"):
            return str(he["tool_mode"])
        return str((self.meta.get("spec") or {}).get("tool_mode") or "normal")

    @property
    def notes(self) -> str:
        return str(self.meta.get("notes") or "")

    @property
    def tags(self) -> list[str]:
        t = self.meta.get("tags")
        return [str(x) for x in t if str(x).strip()] if isinstance(t, list) else []

    @property
    def archived(self) -> bool:
        return bool(self.meta.get("archived"))


def save_meta(run: Run, *, notes: str | None = None, tags: list[str] | None = None,
              archived: bool | None = None) -> None:
    """Persists user annotations (notes, tags, archived) back to `gbtui.json`."""
    if notes is not None:
        run.meta["notes"] = str(notes).strip()
    if tags is not None:
        run.meta["tags"] = [str(t).strip() for t in tags if str(t).strip()]
    if archived is not None:
        run.meta["archived"] = bool(archived)
    p = run.path / "gbtui.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(run.meta, indent=2), encoding="utf-8")
    tmp.replace(p)


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _perf_scores(r: dict) -> list:
    out = []

    def pick(m):
        for k in (f"{m}_p50_ms", f"median_{m}_ms", f"median_{m}_ms_mean", f"mean_{m}_ms", f"mean_{m}_ms_mean"):
            if _num(r.get(k)) is not None:
                return r[k]
        return None
    tpot, ttft = pick("tpot"), pick("ttft")
    n = r.get("completed_requests") or r.get("pooled_sample_count") or r.get("num_prompts") or 0
    tag = r.get("campaign") or (f"in{r.get('input_length')}" if r.get("input_length") else "")
    sfx = f" ({tag})" if tag and tag != "-" else ""
    if tpot:
        out.append(Score("perf", "decode tok/s" + sfx, round(1000.0 / tpot, 2), "tok/s", int(n or 0)))
    if ttft is not None:
        out.append(Score("perf", "TTFT" + sfx, round(ttft / 1000.0, 3), "s", int(n or 0)))
    tp = _num(r.get("output_tokens_per_second_mean")) or _num(r.get("output_throughput"))
    if tp:
        out.append(Score("perf", "output tok/s" + sfx, round(tp, 2), "tok/s", int(n or 0)))
    return out


def load_run(path: Path) -> Run | None:
    if not path.is_dir():
        return None
    run = Run(id=path.name, path=path, meta=_load_dict(path / "gbtui.json"), metadata=_load_dict(path / "metadata.json"),
              summary=_load_dict(path / "summary.json"))
    seen_eval = set()
    results = list(run.summary.get("models") or []) + list(run.summary.get("failures") or [])
    for r in results:
        if not isinstance(r, dict):
            continue
        bt = r.get("benchmark_type")
        if bt == "golden":
            acc = _num(r.get("accuracy_percent"))
            run.scores.append(Score("golden", "golden_set", acc, "%", int(r.get("applicable_tasks") or 0),
                                    int(r.get("passed_cases") or 0), r.get("status", ""),
                                    detail=f"{r.get('passed_cases', 0)} passed · {r.get('failed_cases', 0)} failed · "
                                           f"{r.get('error_cases', 0)} errors · {r.get('not_applicable_cases', 0)} N/A"))
        elif bt == "eval":
            name = r.get("eval_name", "?")
            seen_eval.add(name)
            acc = _num(r.get("accuracy"))
            run.scores.append(Score("eval", name, acc, "%", int(r.get("total_questions") or 0),
                                    _num(r.get("correct_answers")), r.get("status", "failed" if r.get("failed") else ""),
                                    pillar=r.get("pillar", ""), comparable=r.get("leaderboard_comparable"),
                                    detail=r.get("failure_reason") or r.get("error") or r.get("accuracy_withheld_reason") or "",
                                    file=(r.get("sample_traces_omitted") or {}).get("file", "")))
        elif bt == "harness":
            h = r.get("harness") or "harness"
            tm = r.get("tool_mode") or "normal"
            ds = r.get("dataset_short") or r.get("dataset") or "harbor"
            res_rate = _num(r.get("resolved_rate"))
            total_t = int(r.get("total_tasks") or 0)
            res_t = int(r.get("resolved_tasks") or 0)
            avg_tok = _num(r.get("avg_tokens"))
            avg_tc = _num(r.get("avg_tool_calls"))
            avg_cm = _num(r.get("avg_code_mode_calls"))
            avg_nm = _num(r.get("avg_normal_tool_calls"))
            cache_hit = _num(r.get("cache_hit_rate_pct"))
            det_parts = [f"{res_t}/{total_t} resolved"]
            if tm == "code_mode":
                det_parts.append("code-mode")
            if avg_tok is not None:
                det_parts.append(f"avg {avg_tok:.0f} tok")
            if avg_tc is not None:
                det_parts.append(f"avg {avg_tc:.1f} tools")
            if avg_cm is not None and avg_nm is not None:
                det_parts.append(f"code:{avg_cm:.1f}/norm:{avg_nm:.1f}")
            if cache_hit is not None:
                det_parts.append(f"cache {cache_hit:.0f}%")
            score_name = f"{h}[code]:{ds}" if tm == "code_mode" else f"{h}:{ds}"
            run.scores.append(Score("harness", score_name, res_rate, "%", total_t, res_t,
                                    r.get("status", ""), pillar="Harness",
                                    detail=" · ".join(det_parts), file=r.get("trials_file", "harbor_trials.json")))
            if avg_tok is not None:
                run.scores.append(Score("perf", "avg tokens", avg_tok, "tok", total_t))
            if avg_tc is not None:
                run.scores.append(Score("perf", "avg tool calls", avg_tc, "tools", total_t))
            if avg_cm is not None and (avg_cm > 0 or tm == "code_mode"):
                run.scores.append(Score("perf", "code-mode calls", avg_cm, "tools", total_t))
            if avg_nm is not None and (avg_nm > 0 or avg_cm):
                run.scores.append(Score("perf", "normal tool calls", avg_nm, "tools", total_t))
            if cache_hit is not None:
                run.scores.append(Score("perf", "KV cache hit", cache_hit, "pct", total_t))
        elif bt in ("serving", "serving_multimodal", "throughput"):
            run.scores.extend(_perf_scores(r))
        elif bt == "quality":
            pr = _num(r.get("pass_rate"))
            val = pr if pr is not None else _num(r.get("score"))
            run.scores.append(Score("quality", "gemmaclaw_qa", val, "%",
                                    int(r.get("total") or 0), status=r.get("status", "")))
    # eval_*.json not yet in a summary (an interrupted run)
    for f in sorted(path.glob("eval_*.json")):
        d = _load(f)
        if not isinstance(d, dict) or d.get("eval_name") in seen_eval:
            continue
        run.scores.append(Score("eval", d.get("eval_name", f.stem), _num(d.get("accuracy")), "%",
                                int(d.get("total_questions") or 0), _num(d.get("correct_answers")),
                                d.get("status", ""), comparable=d.get("leaderboard_comparable"), file=f.name))
    for s in run.scores:
        if s.kind == "eval" and not s.file:
            m = next(iter(sorted(path.glob(f"eval_{s.name}_*.json"))), None)
            s.file = m.name if m else ""
    return run


def list_runs(results_dir: Path, include_archived: bool = False) -> list:
    runs = []
    for d in sorted(results_dir.iterdir() if results_dir.is_dir() else [], reverse=True):
        r = load_run(d)
        if r and (r.meta or r.summary):
            if include_archived or not r.archived:
                runs.append(r)
    runs.sort(key=lambda r: r.started, reverse=True)
    return runs


# ------------------------------------------------------------------------------------------------
# per-sample items (failure explorer, dataset export)
# ------------------------------------------------------------------------------------------------
def _msgs_prompt(messages) -> str:
    if not isinstance(messages, list):
        return str(messages or "")
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else m
        if isinstance(c, list):
            c = " ".join(p.get("text", "[" + str(p.get("type")) + "]") for p in c if isinstance(p, dict))
        out.append(f"[{m.get('role', '?') if isinstance(m, dict) else '?'}] {c}")
    return "\n".join(out)


def items(run: Run, score: Score | None = None) -> list:
    """Per-sample records: {suite, kind, id, category, prompt, messages, response, reasoning, tool_calls, gold, correct,
    status, detail, finish_reason, completion_tokens}."""
    out = []
    want_golden = score is None or score.kind == "golden"
    if want_golden:
        for r in (run.summary.get("models") or []) + (run.summary.get("failures") or []):
            if not isinstance(r, dict) or r.get("benchmark_type") != "golden":
                continue
            for t in r.get("task_results") or []:
                # gbench keeps the verdict, not the case: prompt and expected answer come from the task's JSON
                task = catalog.golden_task(t.get("task_id") or "") or {}
                msgs = (task.get("conversation") or {}).get("messages")
                truth = task.get("golden_truth") or {}
                gold = next((truth[k] for k in ("canonical_answer", "expected_outputs", "expected_tool_calls")
                             if k in truth and truth[k] is not None), t.get("match_type") or "")
                if not isinstance(gold, str):
                    gold = json.dumps(gold, ensure_ascii=False)
                na = t.get("status") == "not_applicable"
                out.append({"suite": "golden_set", "kind": "golden", "id": t.get("task_id"), "category": t.get("category"),
                            "title": t.get("title") or "", "prompt": _msgs_prompt(msgs) if msgs else "", "messages": msgs,
                            "response": t.get("response_text") or "", "reasoning": t.get("reasoning") or "",
                            "tool_calls": t.get("tool_calls") or [], "gold": gold,
                            "correct": None if na else t.get("status") == "passed", "status": t.get("status"),
                            "detail": t.get("details") or "", "finish_reason": t.get("finish_reason"),
                            "completion_tokens": t.get("completion_tokens")})
    evals = [s for s in run.scores if s.kind == "eval"] if score is None else ([score] if score.kind == "eval" else [])
    for s in evals:
        d = None
        if s.file:
            p_cand = run.path / s.file if not Path(s.file).is_absolute() else run.path / Path(s.file).name
            d = _load(p_cand) or _load(run.path / Path(s.file).name)
        for t in (d if isinstance(d, dict) else {}).get("sample_traces", []) or []:
            out.append({"suite": s.name, "kind": "eval", "id": t.get("sample_key") or t.get("sample_idx"),
                        "category": t.get("category"), "title": "", "prompt": _msgs_prompt(t.get("messages")),
                        "messages": t.get("messages"), "response": t.get("response_text") or "",
                        "reasoning": t.get("reasoning") or "", "tool_calls": t.get("tool_calls") or [],
                        "gold": t.get("gold_answer"), "correct": bool(t.get("is_correct")), "status": t.get("status"),
                        "detail": t.get("error") or t.get("health") or "", "finish_reason": t.get("finish_reason"),
                        "completion_tokens": t.get("completion_tokens")})
    harnesses = [s for s in run.scores if s.kind == "harness"] if score is None else ([score] if score.kind == "harness" else [])
    for s in harnesses:
        tf = run.path / (s.file or "harbor_trials.json")
        trials = _load(tf)
        if not isinstance(trials, list):
            continue
        for t in trials:
            if not isinstance(t, dict):
                continue
            det = []
            if t.get("reward") is not None:
                det.append(f"reward={t['reward']}")
            if t.get("total_tokens") is not None:
                det.append(f"tokens={t['total_tokens']}")
            if t.get("tool_calls") is not None:
                det.append(f"tool_calls={t['tool_calls']}")
            if t.get("duration_s") is not None:
                det.append(f"{t['duration_s']:.1f}s")
            if t.get("error"):
                det.append(str(t["error"]))
            out.append({
                "suite": s.name, "kind": "harness", "id": t.get("task_name"), "category": run.eval_harness or "harness",
                "title": t.get("trial_name") or "", "prompt": t.get("prompt") or "", "messages": None,
                "response": t.get("response") or "", "reasoning": "",
                "tool_calls": t.get("tool_call_list") or [], "gold": "reward >= 1.0",
                "correct": bool(t.get("correct")), "status": t.get("status") or "",
                "detail": " · ".join(det), "finish_reason": None,
                "completion_tokens": t.get("output_tokens") or t.get("total_tokens"),
            })
    return out
