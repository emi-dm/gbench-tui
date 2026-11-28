"""Export runs as evaluation datasets or standardized Knowledge-Distillation / LoRA SFT corpora.

Supports three standard schemas:
  - `chatml`   (default for distillation): OpenAI / Hugging Face TRL `SFTTrainer` / Unsloth Studio & Desktop
               `{"messages": [{"role": ..., "content": ..., "tool_calls": ...}], "metadata": {...}}` in `data/train.jsonl`.
               Reconstructs full multi-turn agent trajectories (system, user, assistant reasoning + tool_calls, tool
               observations) directly from Harbor ATIF `trajectory.json` files when available.
  - `sharegpt`: `{"conversations": [{"from": "human"|"gpt"|"function_call"|"observation"|"system", "value": ...}]}`
               compatible with `unsloth.standardize_sharegpt()` and LLaMA-Factory.
  - `gbench`:  raw tabular evaluation records (`data/samples.jsonl` + `data/scores.jsonl`) with every verdict column.
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

SCHEMAS = ("gbench", "chatml", "sharegpt")
FILTER_MODES = ("all", "passed", "failed")
REASONING_MODES = ("think_tags", "reasoning_content", "strip")

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
suite), run with gbench-tui: every run inside its own hardened Docker container, against an OpenAI-compatible endpoint.

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

DISTILL_CARD = """---
pretty_name: {yaml_title}
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.{ext}
- config_name: scores
  data_files: data/scores.{ext}
tags:
- sft
- knowledge-distillation
- lora
- unsloth
- trl
- {schema}
- gbench
---

# {title}

Standardized **{schema_label}** knowledge-distillation dataset exported by **gbench-tui** (`gbt distill`) for LoRA /
QLoRA fine-tuning in **Unsloth Studio (Desktop)**, **Unsloth + Hugging Face TRL (`SFTTrainer`)**, **MLX-LM**, and
**Axolotl**.

- **Schema**: `{schema}` · **Filter**: `{split_filter}` · **Reasoning**: `{reasoning_mode}`
- **Teacher runs**: {n_runs} · **Teacher model(s)**: {models} · **Training trajectories**: {n_samples} · exported {when}

## Quick Start — Unsloth Studio / Desktop

Drag and drop `data/train.jsonl` directly into **Unsloth Studio / Desktop** and select **{unsloth_format}** format.

## Quick Start — Unsloth + Hugging Face TRL (`SFTTrainer`)

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel{unsloth_extra_import}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B",   # replace with your student base model
    max_seq_length=32768,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

dataset = load_dataset("json", data_files="data/train.jsonl", split="train"){unsloth_norm_line}

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./lora-out",
        max_length=32768,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        assistant_only_loss=True,
    ),
)
trainer.train()
model.save_pretrained_gguf("./lora-gguf", tokenizer, quantization_method="f16")
```

## License & Provenance

Verify the licenses of the underlying evaluation prompts and teacher models before publishing distilled weights or
datasets.
"""


def _format_assistant_text(response: str, reasoning: str, reasoning_mode: str) -> tuple[str, str]:
    """Returns `(content, reasoning_content)` according to `reasoning_mode`."""
    resp = (response or "").strip()
    reas = (reasoning or "").strip()
    if reasoning_mode == "strip" or not reas:
        return resp, ""
    if reasoning_mode == "reasoning_content":
        return resp, reas
    # Default: think_tags (Unsloth Studio / DeepSeek-R1 / Qwen3 / LFM3 compatible)
    if "<think>" in resp:
        return resp, ""
    content = f"<think>\n{reas}\n</think>\n\n{resp}" if resp else f"<think>\n{reas}\n</think>"
    return content, ""


def _normalize_tool_calls(raw_calls) -> list[dict]:
    """Converts gbench / ATIF / OpenAI tool calls into canonical OpenAI `tool_calls` format."""
    out = []
    if not isinstance(raw_calls, list):
        return out
    for idx, tc in enumerate(raw_calls):
        if isinstance(tc, str):
            out.append({
                "id": f"call_{idx + 1}",
                "type": "function",
                "function": {"name": tc, "arguments": "{}"},
            })
            continue
        if not isinstance(tc, dict):
            continue
        if "function" in tc and isinstance(tc["function"], dict):
            fn = tc["function"]
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False, default=str)
            out.append({
                "id": str(tc.get("id") or f"call_{idx + 1}"),
                "type": "function",
                "function": {"name": str(fn.get("name") or "unknown"), "arguments": args},
            })
            continue
        name = tc.get("function_name") or tc.get("name") or tc.get("tool_name") or "unknown"
        args = tc.get("arguments") if "arguments" in tc else tc.get("input", {})
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False, default=str)
        out.append({
            "id": str(tc.get("tool_call_id") or tc.get("id") or f"call_{idx + 1}"),
            "type": "function",
            "function": {"name": str(name), "arguments": args},
        })
    return out


def _flatten_content(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or f"[{p.get('type', 'part')}]"))
        return "\n".join(x for x in parts if x)
    return json.dumps(c, ensure_ascii=False, default=str)


def _find_harbor_trajectory(run: R.Run, trial_name: str) -> Path | None:
    """Locate the ATIF `trajectory.json` for a Harbor trial (preserved in `trajectories/` or Harbor jobs cache)."""
    if not trial_name:
        return None
    candidates = [
        run.path / "trajectories" / f"{trial_name}.json",
        run.path / f"{trial_name}.json",
        run.path / trial_name / "agent" / "trajectory.json",
        sandbox.work_dir() / "harbor-jobs" / run.id / trial_name / "agent" / "trajectory.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def harbor_trajectory_messages(traj_path: Path, reasoning_mode: str = "think_tags") -> list[dict]:
    """Reconstructs a full multi-turn OpenAI/TRL `messages` list from an ATIF `trajectory.json`."""
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    messages: list[dict] = []
    last_call_ids: list[tuple[str, str]] = []
    for st in data.get("steps") or []:
        if not isinstance(st, dict):
            continue
        src = str(st.get("source") or "").lower()
        msg_txt = _flatten_content(st.get("message"))
        if src == "system":
            if msg_txt.strip():
                messages.append({"role": "system", "content": msg_txt.strip()})
        elif src == "user":
            if msg_txt.strip():
                messages.append({"role": "user", "content": msg_txt.strip()})
        elif src in ("agent", "assistant"):
            reas = str(st.get("reasoning_content") or st.get("reasoning") or "")
            content, reas_out = _format_assistant_text(msg_txt, reas, reasoning_mode)
            tcs = _normalize_tool_calls(st.get("tool_calls") or [])
            if not content and not reas_out and not tcs:
                continue
            m: dict = {"role": "assistant", "content": content}
            if reas_out:
                m["reasoning_content"] = reas_out
            if tcs:
                m["tool_calls"] = tcs
                last_call_ids = [(tc["id"], tc["function"]["name"]) for tc in tcs]
            else:
                last_call_ids = []
            messages.append(m)
            obs = st.get("observation")
            if isinstance(obs, dict):
                for idx, res in enumerate(obs.get("results") or []):
                    if not isinstance(res, dict):
                        continue
                    cid = str(res.get("source_call_id") or (last_call_ids[idx][0] if idx < len(last_call_ids) else f"call_{idx + 1}"))
                    cname = last_call_ids[idx][1] if idx < len(last_call_ids) else ""
                    obs_msg: dict = {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": _flatten_content(res.get("content")),
                    }
                    if cname:
                        obs_msg["name"] = cname
                    messages.append(obs_msg)
        elif src in ("observation", "tool"):
            obs = st.get("observation")
            results_list = (obs.get("results") if isinstance(obs, dict) else None) or []
            if results_list:
                for idx, res in enumerate(results_list):
                    if not isinstance(res, dict):
                        continue
                    cid = str(res.get("source_call_id") or (last_call_ids[idx][0] if idx < len(last_call_ids) else f"call_{idx + 1}"))
                    cname = last_call_ids[idx][1] if idx < len(last_call_ids) else ""
                    obs_msg = {"role": "tool", "tool_call_id": cid, "content": _flatten_content(res.get("content"))}
                    if cname:
                        obs_msg["name"] = cname
                    messages.append(obs_msg)
            elif msg_txt.strip():
                messages.append({"role": "tool", "tool_call_id": last_call_ids[0][0] if last_call_ids else "call_1", "content": msg_txt.strip()})
    return messages


def item_to_messages(it: dict, run: R.Run, reasoning_mode: str = "think_tags",
                     include_harbor_trajectories: bool = True) -> list[dict]:
    """Converts any gbench or Harbor sample item into standard OpenAI / HF TRL `messages`."""
    if include_harbor_trajectories and it.get("kind") == "harness":
        traj = _find_harbor_trajectory(run, str(it.get("title") or ""))
        if traj:
            msgs = harbor_trajectory_messages(traj, reasoning_mode=reasoning_mode)
            if any(m.get("role") == "assistant" for m in msgs):
                return msgs

    messages: list[dict] = []
    raw_msgs = it.get("messages")
    if isinstance(raw_msgs, list) and raw_msgs:
        for m in raw_msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "user")
            content = _flatten_content(m.get("content"))
            messages.append({"role": role, "content": content})
    else:
        prompt = str(it.get("prompt") or it.get("title") or "").strip()
        if prompt:
            messages.append({"role": "user", "content": prompt})

    content, reas_out = _format_assistant_text(str(it.get("response") or ""), str(it.get("reasoning") or ""), reasoning_mode)
    tcs = _normalize_tool_calls(it.get("tool_calls") or [])
    if not content and not reas_out and not tcs:
        return []
    asst: dict = {"role": "assistant", "content": content}
    if reas_out:
        asst["reasoning_content"] = reas_out
    if tcs:
        asst["tool_calls"] = tcs
    # Avoid duplicating trailing assistant turn if already present in `raw_msgs`
    if messages and messages[-1].get("role") == "assistant":
        messages[-1] = asst
    else:
        messages.append(asst)
    return messages


def messages_to_sharegpt(messages: list[dict]) -> list[dict]:
    """Converts OpenAI/ChatML `messages` (including tool_calls & reasoning) to ShareGPT `conversations`."""
    role_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "observation"}
    convs: list[dict] = []
    for m in messages:
        role = str(m.get("role") or "user")
        val = str(m.get("content") or "")
        if m.get("reasoning_content"):
            val = f"<think>\n{m['reasoning_content']}\n</think>\n\n{val}" if val else f"<think>\n{m['reasoning_content']}\n</think>"
        if role == "assistant" and m.get("tool_calls"):
            if val.strip():
                convs.append({"from": "gpt", "value": val})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                convs.append({
                    "from": "function_call",
                    "value": json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments")}, ensure_ascii=False),
                })
        else:
            convs.append({"from": role_map.get(role, role), "value": val})
    return convs


def build_distill_record(
    it: dict,
    run: R.Run,
    schema: str = "chatml",
    reasoning_mode: str = "think_tags",
    include_harbor_trajectories: bool = True,
) -> dict | None:
    """Builds a standardized SFT / distillation training example (`chatml` or `sharegpt`), or None if empty."""
    msgs = item_to_messages(it, run, reasoning_mode=reasoning_mode,
                            include_harbor_trajectories=include_harbor_trajectories)
    if not msgs or not any(m.get("role") == "assistant" for m in msgs):
        return None
    meta = {
        "run_id": run.id,
        "teacher_model": run.model_label,
        "base_model": run.base_model,
        "quant": run.quant,
        "adapter": run.adapter or "base",
        "harness": run.eval_harness,
        "tool_mode": run.tool_mode,
        "suite": it.get("suite"),
        "sample_id": str(it.get("id")) if it.get("id") is not None else None,
        "category": it.get("category"),
        "correct": it.get("correct"),
        "completion_tokens": it.get("completion_tokens"),
    }
    if schema == "sharegpt":
        return {"conversations": messages_to_sharegpt(msgs), "metadata": meta}
    return {"messages": msgs, "metadata": meta}


def export(
    runs: list,
    out: Path,
    fmt: str = "jsonl",
    only_failures: bool = False,
    on_line=print,
    *,
    schema: str = "gbench",
    split_filter: str = "all",
    reasoning_mode: str = "think_tags",
    include_harbor_trajectories: bool = True,
) -> Path:
    """Exports runs as an evaluation dataset (`schema='gbench'`) or a standardized distillation / LoRA SFT dataset
    (`schema='chatml'` for Unsloth Studio & HF TRL, or `schema='sharegpt'`).

    `fmt`: jsonl | parquet | both. Returns the output folder.
    """
    schema = schema if schema in SCHEMAS else "gbench"
    if only_failures and split_filter == "all":
        split_filter = "failed"
    split_filter = split_filter if split_filter in FILTER_MODES else "all"
    reasoning_mode = reasoning_mode if reasoning_mode in REASONING_MODES else "think_tags"

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    is_distill = schema in ("chatml", "sharegpt")
    primary_name = "train" if is_distill else "samples"
    n_samples = 0
    models = set()

    with open(data / f"{primary_name}.jsonl", "w", encoding="utf-8") as fs, open(data / "scores.jsonl", "w", encoding="utf-8") as fc:
        for run in runs:
            models.add(run.model_label)
            spec = run.meta.get("spec") or {}
            ctx = {"run_id": run.id, "model": run.model, "base_model": run.base_model, "quant": run.quant,
                   "adapter": run.adapter or "base", "target_harness": run.target_harness, "harness": run.eval_harness,
                   "tool_mode": run.tool_mode,
                   "served_model": run.endpoint.get("model", ""),
                   "tokenizer": run.endpoint.get("tokenizer", ""), "tier": run.tier, "date": run.when,
                   "thinking": bool(spec.get("thinking")), "max_output_tokens": spec.get("max_output_tokens"),
                   "eval_limit": spec.get("eval_limit") if run.tier in ("evals", "harness") else None}
            for s in run.scores:
                fc.write(json.dumps({**ctx, "kind": s.kind, "benchmark": s.name, "value": s.value, "unit": s.unit,
                                     "n": s.n, "correct": s.correct, "status": s.status,
                                     "leaderboard_comparable": s.comparable}, default=str) + "\n")
            for it in R.items(run):
                if split_filter == "failed" and it["correct"] is not False:
                    continue
                if split_filter == "passed" and it["correct"] is not True:
                    continue
                if is_distill:
                    rec = build_distill_record(
                        it, run, schema=schema, reasoning_mode=reasoning_mode,
                        include_harbor_trajectories=include_harbor_trajectories,
                    )
                    if not rec:
                        continue
                    fs.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    n_samples += 1
                else:
                    gold_val = it["gold"]
                    if gold_val is not None and not isinstance(gold_val, str):
                        gold_val = json.dumps(gold_val, default=str)
                    row = {**ctx, **{k: it[k] for k in ("suite", "kind", "category", "title", "prompt", "response", "reasoning",
                                                         "finish_reason", "completion_tokens", "status", "detail", "correct")},
                           "sample_id": str(it["id"]) if it["id"] is not None else None,
                           "messages": json.dumps(it["messages"], default=str) if it["messages"] is not None else None,
                           "tool_calls": json.dumps(it["tool_calls"], default=str) if it["tool_calls"] else None,
                           "gold": gold_val}
                    fs.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    n_samples += 1

    on_line(f"✔ {n_samples} {'training trajectories' if is_distill else 'samples'} ({schema}) and scores of {len(runs)} run(s) written to {data}")
    ext = "jsonl"
    if fmt in ("parquet", "both"):
        if _to_parquet(data, (primary_name, "scores"), on_line):
            ext = "parquet"
            if fmt == "parquet":
                for f in (f"{primary_name}.jsonl", "scores.jsonl"):
                    (data / f).unlink(missing_ok=True)
    commit = next((r.meta.get("gbench_commit") for r in runs if r.meta.get("gbench_commit")), "") or "unknown"
    title = out.resolve().name or "gbench-dataset"
    if is_distill:
        is_sg = schema == "sharegpt"
        (out / "README.md").write_text(
            DISTILL_CARD.format(
                title=title, yaml_title=json.dumps(title), ext=ext, schema=schema,
                schema_label="ShareGPT (conversations)" if is_sg else "OpenAI / HF TRL ChatML (messages)",
                unsloth_format="ShareGPT" if is_sg else "ChatML / OpenAI Messages",
                unsloth_extra_import=", standardize_sharegpt" if is_sg else "",
                unsloth_norm_line="\ndataset = standardize_sharegpt(dataset)" if is_sg else "",
                split_filter=split_filter, reasoning_mode=reasoning_mode,
                n_runs=len(runs), models=", ".join(sorted(models)), n_samples=n_samples,
                when=time.strftime("%Y-%m-%d %H:%M"),
            ),
            encoding="utf-8",
        )
    else:
        (out / "README.md").write_text(
            CARD.format(title=title, yaml_title=json.dumps(title), ext=ext, n_runs=len(runs),
                        models=", ".join(sorted(models)), n_samples=n_samples,
                        when=time.strftime("%Y-%m-%d %H:%M"), commit=commit),
            encoding="utf-8",
        )
    on_line(f"✔ dataset card: {out / 'README.md'}")
    return out


def _to_parquet(data: Path, names: tuple[str, ...], on_line) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.json as pj                    # noqa: F401
        import pyarrow.parquet as pq
        for name in names:
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
    if sandbox.status():
        on_line("⚠ parquet needs pyarrow on the host or the Docker image: kept JSONL")
        return False
    stage = sandbox.work_dir() / f"export-{int(time.time())}-{secrets.token_hex(3)}"
    box = ""
    try:
        stage.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy(data / f"{name}.jsonl", stage / f"{name}.jsonl")
        on_line("converting to parquet inside an offline Docker container (pyarrow from gbench's environment)…")
        box = sandbox.create(2, 2, [], on_line)
        work = sandbox.CONTAINER_WORK
        names_json = json.dumps(list(names))
        code = ("import json,os,pyarrow as pa,pyarrow.json as pj,pyarrow.parquet as pq\n"
                f"for n in json.loads({names_json!r}):\n"
                f"    src = f'{work}/{stage.name}/{{n}}.jsonl'\n"
                f"    dst = f'{work}/{stage.name}/{{n}}.parquet'\n"
                "    tbl = pa.table({}) if os.path.getsize(src) == 0 else pj.read_json(src)\n"
                "    pq.write_table(tbl, dst)\n")
        r = sandbox.run_exec(box, [f"{sandbox.VENV}/bin/python", "-c", code], timeout=600)
        if r.returncode != 0:
            on_line("⚠ parquet conversion failed: " + (r.stderr or r.stdout)[-200:])
            return False
        for name in names:
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
