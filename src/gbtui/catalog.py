"""What gbench can run: eval suites by pillar, the Golden Set, and each suite's own documentation.

Two sources, both gbench's own: the catalog dumped by gbench inside the sandbox when the template is built
(~/.config/gbench-tui/catalog.json: suites, pillars, [tools] flag, Golden Set tasks) and the docs of the pinned
checkout (docs/eval-suites.md table for titles and descriptions, docs/evals/<suite>.md for the full card).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .config import cache_dir, config_dir

# Suites that work on a laptop against a small local model: no Docker image to build, no judge key, short prompts.
LIGHT = ["gsm8k", "ifeval", "mmlu_pro", "gpqa_diamond", "causalbench", "arc_agi", "imo_answer_bench", "mmlu_redux",
         "multilingual_mmlu", "cruxeval"]


def src_dir() -> Path:
    return cache_dir() / "gbench-src"


@lru_cache(maxsize=1)
def _table() -> dict:
    """docs/eval-suites.md: {suite: {'pillar', 'title', 'description', 'options'}}"""
    p = src_dir() / "docs" / "eval-suites.md"
    out, pillar = {}, ""
    try:
        text = p.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if not line.startswith("|") or "Suite ID" in line or line.startswith("| :"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        m = re.match(r"\*\*\d+\.\s*(.+?)\*\*", cells[0])
        if m:
            pillar = m.group(1)
        sid = re.search(r"`([a-z0-9_]+)`", cells[1])
        if not sid:
            continue
        desc = re.sub(r"\(\[Setup Guide\]\([^)]*\)\)", "", cells[3]).strip()
        desc = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", desc)
        out[sid.group(1)] = {"pillar": pillar, "title": cells[2], "description": desc,
                             "options": cells[4] if len(cells) > 4 else ""}
    return out


def needs(name: str, desc: str = "") -> list[str]:
    """What a suite needs beyond an endpoint (read from its description)."""
    d = (desc or _table().get(name, {}).get("description", "")).lower()
    req = []
    if "local image" in d or "docker" in d or "local `gbench-" in d or name in (
            "bfcl", "bfcl_live", "bfcl_v4_agentic", "tau2", "tau3", "terminal_bench", "aider_polyglot", "multipl_e",
            "ojbench", "bigcodebench", "aime", "hmmt", "swe_bench_pro", "swe_bench_live", "multi_swe_bench",
            "copilot_bench_swe", "swe_bench_multilingual", "swe_lancer", "gaia2", "mcp_bench", "mcp_atlas",
            "cyberseceval", "putnam_formal", "skillsbench", "toolbench", "livebench", "ui_control_osworld", "mobilegym"):
        req.append("docker image")
    if "gemini" in d or "judge" in d or name in ("simpleqa", "frames", "browsecomp", "healthbench", "gdpval",
                                                 "humanitys_last_exam", "wildbench", "cimemories"):
        req.append("Gemini key")
    if "logprob" in d or name in ("mmlu", "wmdp"):
        req.append("logprobs")
    if "128k" in d or "1m tokens" in d or "100k" in d:
        req.append("long context")
    if "image" in d and ("vision" in d or "screenshot" in d or "mm " in d or "multimodal" in d) or name in (
            "mmmu_pro", "screenspot", "textvqa", "chartqa", "docvqa", "mathvista", "coco_caption", "omnidocbench"):
        req.append("vision")
    return req


@lru_cache(maxsize=1)
def load() -> dict:
    """{'suites': [{name, pillar, title, description, tools, named_only, needs, light}], 'pillars': [...],
    'golden': [...], 'gbench_version', 'source'}"""
    table = _table()
    try:
        data = json.loads((config_dir() / "catalog.json").read_text())
        source = "gbench (inside the sandbox)"
    except (OSError, ValueError):
        data = {"suites": [{"name": k, "pillar": v["pillar"], "tools": False, "named_only": False}
                           for k, v in table.items()],
                "pillars": list(dict.fromkeys(v["pillar"] for v in table.values())), "golden": _golden_from_src()}
        source = "gbench docs (build the sandbox template for gbench's own list)"
    for s in data["suites"]:
        t = table.get(s["name"], {})
        s.setdefault("title", t.get("title") or s["name"])
        s["title"] = t.get("title") or s["name"]
        s["description"] = t.get("description", "")
        s["options"] = t.get("options", "")
        if not s.get("pillar") or s["pillar"] == "Other":
            s["pillar"] = t.get("pillar") or "Other"
        s["pillar"] = re.sub(r"^\d+\.\s*", "", s["pillar"])
        s["needs"] = needs(s["name"], s["description"])
        if s.get("tools"):
            s["needs"].append("tool calls")
        s["light"] = s["name"] in LIGHT
    data["pillars"] = [re.sub(r"^\d+\.\s*", "", p) for p in data.get("pillars", [])]
    data["source"] = source
    return data


def _golden_from_src() -> list:
    out = []
    for fp in sorted((src_dir() / "gbench" / "golden_dataset").glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except (OSError, ValueError):
            continue
        out.append({"id": d.get("id", fp.stem), "category": d.get("category", ""),
                    "description": d.get("description") or d.get("title") or d.get("name", ""), "file": fp.name})
    return out


def suite(name: str) -> dict | None:
    return next((s for s in load()["suites"] if s["name"] == name), None)


def doc(name: str) -> str:
    """The suite's setup and scoring guide (docs/evals/<name>.md), or its table row."""
    p = src_dir() / "docs" / "evals" / f"{name}.md"
    try:
        return p.read_text()
    except OSError:
        s = suite(name) or {}
        return f"# {s.get('title', name)}\n\n{s.get('description', '(no description)')}\n"


def golden_task(task_id: str) -> dict | None:
    for fp in (src_dir() / "gbench" / "golden_dataset").glob("*.json"):
        try:
            d = json.loads(fp.read_text())
        except (OSError, ValueError):
            continue
        if d.get("id", fp.stem) == task_id:
            return d
    return None


def reload() -> None:
    load.cache_clear()
    _table.cache_clear()
