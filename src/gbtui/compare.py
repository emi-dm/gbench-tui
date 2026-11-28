"""Deep diagnostics & side-by-side Run A vs Run B comparison (regressions, improvements, failure taxonomy, Token Economy)."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import results as R

_TOOL_FORMAT_PATTERNS = ("json", "tool", "schema", "format", "parse", "function", "syntax")
_TIMEOUT_ERROR_PATTERNS = ("timeout", "timed out", "connection", "http ", "exception", "error", "traceback", "oom")


def classify_failure(item: dict) -> str:
    """Classifies a failed sample into: 'truncated' | 'tool_format' | 'timeout_error' | 'wrong_answer'."""
    if str(item.get("finish_reason") or "").lower() == "length":
        return "truncated"
    detail = str(item.get("detail") or "").lower()
    status = str(item.get("status") or "").lower()
    cat = str(item.get("category") or "").lower()
    if status == "error" or any(p in detail for p in _TIMEOUT_ERROR_PATTERNS):
        return "timeout_error"
    if any(p in detail for p in _TOOL_FORMAT_PATTERNS) or cat in ("structured_output", "tool_use"):
        return "tool_format"
    return "wrong_answer"


def diagnose_run(run: R.Run) -> dict:
    """Computes failure-cause breakdown and Token Economy metrics for a run."""
    its = R.items(run)
    causes = {"truncated": 0, "tool_format": 0, "timeout_error": 0, "wrong_answer": 0}
    pass_toks: list[int] = []
    fail_toks: list[int] = []
    total_pass = total_fail = total_na = 0
    for it in its:
        ok = it.get("correct")
        tok = it.get("completion_tokens")
        tok_i = int(tok) if isinstance(tok, (int, float)) and tok >= 0 else None
        if ok is None:
            total_na += 1
        elif ok:
            total_pass += 1
            if tok_i is not None:
                pass_toks.append(tok_i)
        else:
            total_fail += 1
            causes[classify_failure(it)] += 1
            if tok_i is not None:
                fail_toks.append(tok_i)
    all_toks = pass_toks + fail_toks
    total_tok = sum(all_toks)
    avg_pass = round(sum(pass_toks) / len(pass_toks), 1) if pass_toks else None
    avg_fail = round(sum(fail_toks) / len(fail_toks), 1) if fail_toks else None
    avg_all = round(total_tok / len(all_toks), 1) if all_toks else None
    pct_scores = [s.value for s in run.scores if s.unit == "%" and s.value is not None]
    gscore = (sum(pct_scores) / len(pct_scores)) if pct_scores else None
    g_per_1k = round((gscore / (avg_all / 1000.0)), 2) if (gscore is not None and avg_all and avg_all > 0) else None
    return {
        "total": len(its),
        "passed": total_pass,
        "failed": total_fail,
        "na": total_na,
        "causes": causes,
        "avg_tokens_pass": avg_pass,
        "avg_tokens_fail": avg_fail,
        "avg_tokens": avg_all,
        "total_tokens": total_tok,
        "gscore": round(gscore, 2) if gscore is not None else None,
        "gscore_per_1k_tokens": g_per_1k,
    }


@dataclass
class SampleDiff:
    suite: str
    sample_id: str
    category: str
    verdict: str            # regression | improvement | both_pass | both_fail | other
    a: dict | None
    b: dict | None
    token_delta: int | None = None


@dataclass
class RunComparison:
    run_a: R.Run
    run_b: R.Run
    score_deltas: list[dict] = field(default_factory=list)
    samples: list[SampleDiff] = field(default_factory=list)
    regressions: int = 0
    improvements: int = 0
    both_pass: int = 0
    both_fail: int = 0
    other: int = 0
    diag_a: dict = field(default_factory=dict)
    diag_b: dict = field(default_factory=dict)


def compare_runs(run_a: R.Run, run_b: R.Run) -> RunComparison:
    """Aligns scores and per-sample verdicts between Run A (baseline) and Run B (candidate)."""
    cmp = RunComparison(run_a=run_a, run_b=run_b, diag_a=diagnose_run(run_a), diag_b=diagnose_run(run_b))
    map_a = {s.name: s for s in run_a.scores}
    map_b = {s.name: s for s in run_b.scores}
    for name in sorted(set(map_a) | set(map_b)):
        sa, sb = map_a.get(name), map_b.get(name)
        va = sa.value if sa else None
        vb = sb.value if sb else None
        delta = (vb - va) if (va is not None and vb is not None) else None
        unit = (sa or sb).unit if (sa or sb) else ""
        cmp.score_deltas.append({"name": name, "unit": unit, "a": va, "b": vb, "delta": delta})

    its_a = {(str(it.get("suite")), str(it.get("id"))): it for it in R.items(run_a)}
    its_b = {(str(it.get("suite")), str(it.get("id"))): it for it in R.items(run_b)}
    keys = list(dict.fromkeys(list(its_a.keys()) + list(its_b.keys())))
    for k in keys:
        ia, ib = its_a.get(k), its_b.get(k)
        ca = ia.get("correct") if ia else None
        cb = ib.get("correct") if ib else None
        if ca is True and cb is False:
            v = "regression"
            cmp.regressions += 1
        elif ca is False and cb is True:
            v = "improvement"
            cmp.improvements += 1
        elif ca is True and cb is True:
            v = "both_pass"
            cmp.both_pass += 1
        elif ca is False and cb is False:
            v = "both_fail"
            cmp.both_fail += 1
        else:
            v = "other"
            cmp.other += 1
        ta = ia.get("completion_tokens") if ia else None
        tb = ib.get("completion_tokens") if ib else None
        td = (int(tb) - int(ta)) if (isinstance(ta, (int, float)) and isinstance(tb, (int, float))) else None
        cat = str((ia or ib or {}).get("category") or "")
        cmp.samples.append(SampleDiff(suite=k[0], sample_id=k[1], category=cat, verdict=v, a=ia, b=ib, token_delta=td))
    # Sort: regressions first, then improvements, then both_fail, then both_pass
    order = {"regression": 0, "improvement": 1, "both_fail": 2, "other": 3, "both_pass": 4}
    cmp.samples.sort(key=lambda s: (order.get(s.verdict, 5), s.suite, s.sample_id))
    return cmp
