"""Export runs as a dataset: a folder ready for `huggingface-cli upload` (or `datasets.load_dataset("json", …)`).

    <out>/
      README.md            dataset card (YAML header: configs, license note, provenance)
      data/samples.jsonl   one row per sample: run, model, suite, prompt, messages, response, reasoning, tool calls,
                           gold answer, correct, status, detail, finish reason, tokens
      data/scores.jsonl    one row per (run, benchmark): the scores gbench reported
      data/*.parquet       the same, when pyarrow is available (on the host, else converted inside a sandbox)
"""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import time
from pathlib import Path

from . import results as R
from . import sandbox

CARD = """---
pretty_name: {yaml_title}
configs:
- config_name: samples
  data_files: data/samples.{ext}
- config_name: scores
  data_files: data/scores.{ext}
tags:
- gbench
- evaluation
- llm
---

# {title}

Evaluation records produced by [gbench](https://github.com/google-deepmind/gbench) (Google DeepMind's benchmarking
suite), run with gbench-tui: every run inside its own Docker Sandbox, against an OpenAI-compatible endpoint.

- **Runs**: {n_runs} · **models**: {models} · **samples**: {n_samples} · exported {when}
- **gbench commit**: {commit}

## Configs

- `samples`: one row per evaluated sample (prompt / messages, model response, reasoning, tool calls, gold answer,
  whether it was correct and why not).
- `scores`: one row per (run, benchmark) with the score gbench reported, the number of samples and the run settings.

## License of the content

The prompts and gold answers come from the public datasets each gbench suite loads; each keeps its own license (see
gbench's docs/THIRD_PARTY_DATASETS.md). Read those terms before redistributing this dataset.
"""


def export(runs: list, out: Path, fmt: str = "jsonl", only_failures: bool = False, on_line=print) -> Path:
    """fmt: jsonl | parquet | both. Returns the output folder."""
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    models = set()
    with open(data / "samples.jsonl", "w", encoding="utf-8") as fs, open(data / "scores.jsonl", "w", encoding="utf-8") as fc:
        for run in runs:
            models.add(run.model_label)
            spec = run.meta.get("spec") or {}
            ctx = {"run_id": run.id, "model": run.model, "base_model": run.base_model, "quant": run.quant,
                   "served_model": run.endpoint.get("model", ""),
                   "tokenizer": run.endpoint.get("tokenizer", ""), "tier": run.tier, "date": run.when,
                   "thinking": bool(spec.get("thinking")), "max_output_tokens": spec.get("max_output_tokens"),
                   "eval_limit": spec.get("eval_limit") if run.tier == "evals" else None}
            for s in run.scores:
                fc.write(json.dumps({**ctx, "kind": s.kind, "benchmark": s.name, "value": s.value, "unit": s.unit,
                                     "n": s.n, "correct": s.correct, "status": s.status,
                                     "leaderboard_comparable": s.comparable}, default=str) + "\n")
            for it in R.items(run):
                if only_failures and it["correct"] is not False:
                    continue
                gold_val = it["gold"]
                if gold_val is not None and not isinstance(gold_val, str):
                    gold_val = json.dumps(gold_val, default=str)
                row = {**ctx, **{k: it[k] for k in ("suite", "kind", "category", "title", "prompt", "response", "reasoning",
                                                     "finish_reason", "completion_tokens", "status", "detail", "correct")},
                       "sample_id": str(it["id"]) if it["id"] is not None else None,
                       "messages": json.dumps(it["messages"], default=str) if it["messages"] is not None else None,
                       "tool_calls": json.dumps(it["tool_calls"], default=str) if it["tool_calls"] else None,
                       "gold": gold_val}
                fs.write(json.dumps(row, default=str) + "\n")
                n_samples += 1
    on_line(f"✔ {n_samples} samples and the scores of {len(runs)} runs written to {data}")
    ext = "jsonl"
    if fmt in ("parquet", "both"):
        if _to_parquet(data, on_line):
            ext = "parquet"
            if fmt == "parquet":
                for f in ("samples.jsonl", "scores.jsonl"):
                    (data / f).unlink(missing_ok=True)
    commit = next((r.meta.get("gbench_commit") for r in runs if r.meta.get("gbench_commit")), "") or "unknown"
    title = out.resolve().name or "gbench-dataset"
    (out / "README.md").write_text(
        CARD.format(title=title, yaml_title=json.dumps(title), ext=ext, n_runs=len(runs),
                    models=", ".join(sorted(models)), n_samples=n_samples,
                    when=time.strftime("%Y-%m-%d %H:%M"), commit=commit),
        encoding="utf-8",
    )
    on_line(f"✔ dataset card: {out / 'README.md'}")
    return out


def _to_parquet(data: Path, on_line) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.json as pj                    # noqa: F401
        import pyarrow.parquet as pq
        for name in ("samples", "scores"):
            src = data / f"{name}.jsonl"
            dst = data / f"{name}.parquet"
            if src.stat().st_size == 0:
                pq.write_table(pa.table({}), str(dst))
            else:
                pq.write_table(pj.read_json(str(src)), str(dst))
        on_line("✔ parquet written (pyarrow on the host)")
        return True
    except ImportError:
        pass
    except Exception as e:                           # noqa: BLE001
        on_line(f"⚠ parquet on the host failed: {e}")
        return False
    # no pyarrow on the host: gbench's sandbox has it
    if sandbox.status():
        on_line("⚠ parquet needs pyarrow on the host or the sandbox template: kept JSONL")
        return False
    stage = sandbox.work_dir() / f"export-{int(time.time())}-{secrets.token_hex(3)}"
    box = ""
    try:
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("samples", "scores"):
            shutil.copy(data / f"{name}.jsonl", stage / f"{name}.jsonl")
        on_line("converting to parquet inside a sandbox (pyarrow from gbench's environment)…")
        box = sandbox.create(2, 2, [], on_line)
        work = str(sandbox.work_dir())
        code = ("import os,pyarrow as pa,pyarrow.json as pj,pyarrow.parquet as pq\n"
                "for n in ('samples','scores'):\n"
                f"    src = f'{work}/{stage.name}/{{n}}.jsonl'\n"
                f"    dst = f'{work}/{stage.name}/{{n}}.parquet'\n"
                "    tbl = pa.table({}) if os.path.getsize(src) == 0 else pj.read_json(src)\n"
                "    pq.write_table(tbl, dst)\n")
        r = sandbox.sbx("exec", box, f"{sandbox.VENV}/bin/python", "-c", code, timeout=600)
        if r.returncode != 0:
            on_line("⚠ parquet conversion failed: " + (r.stderr or r.stdout)[-200:])
            return False
        for name in ("samples", "scores"):
            shutil.move(str(stage / f"{name}.parquet"), str(data / f"{name}.parquet"))
        on_line("✔ parquet written")
        return True
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        on_line(f"⚠ parquet: {e}")
        return False
    finally:
        if box:
            sandbox.destroy(box)
        shutil.rmtree(stage, ignore_errors=True)
