"""Zero-dependency Hugging Face `harness-hub` LoRA browser and downloader (stdlib urllib only).

Queries `https://huggingface.co/api/collections?owner=harness-hub` and
`https://huggingface.co/api/models?author=harness-hub&full=true`, discovers `*-lora-*.gguf` adapters, matches them
against a local base model (e.g. LFM3-1.7B, Qwen3.8-27B), and downloads them atomically into
`~/.cache/gbench-tui/loras/`.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

HF_API = "https://huggingface.co/api"
HF_WEB = "https://huggingface.co"
DEFAULT_AUTHOR = "harness-hub"
_UA = "gbench-tui/0.1 (urllib)"


def loras_dir() -> Path:
    d = config.cache_dir() / "loras"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_json(url: str, timeout: float = 20.0):
    headers = {"User-Agent": _UA, "Accept": "application/json"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _infer_base_from_repo(repo_id: str, tags: list[str]) -> str:
    """Extract canonical base model name from a harness-hub repo id & HF tags."""
    for t in tags:
        if isinstance(t, str) and t.startswith("base_model:") and not t.startswith("base_model:adapter:"):
            cand = t.split(":", 1)[1].strip()
            if "/" in cand:
                owner, name = cand.split("/", 1)
                if owner.lower() != DEFAULT_AUTHOR:
                    return config.strip_adapter_suffix(name, harness=config.detect_harness(name))
    slug = repo_id.rsplit("/", 1)[-1]
    slug = re.sub(r"[-_]?GGUF$", "", slug, flags=re.IGNORECASE)
    h = config.detect_harness(slug)
    return config.strip_adapter_suffix(slug, harness=h)


def fetch_collections(owner: str = DEFAULT_AUTHOR, timeout: float = 15.0) -> list[dict]:
    """Returns collections from `https://huggingface.co/api/collections?owner=<owner>`."""
    url = f"{HF_API}/collections?owner={urllib.parse.quote(owner)}"
    raw = _get_json(url, timeout=timeout)
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        items = [it.get("id") for it in (c.get("items") or []) if isinstance(it, dict) and it.get("id")]
        out.append({
            "slug": str(c.get("slug") or ""),
            "title": str(c.get("title") or ""),
            "description": str(c.get("description") or ""),
            "items": items,
        })
    return out


def parse_gguf_models(raw_models: list[dict]) -> list[dict]:
    """Normalizes HF `/api/models?author=harness-hub&full=true` rows into downloadable GGUF LoRA adapter entries."""
    out: list[dict] = []
    cache = loras_dir()
    for m in raw_models:
        if not isinstance(m, dict):
            continue
        repo_id = str(m.get("id") or m.get("modelId") or "")
        if not repo_id:
            continue
        tags = [str(t) for t in (m.get("tags") or []) if isinstance(t, str)]
        siblings = [str(s.get("rfilename") or "") for s in (m.get("siblings") or []) if isinstance(s, dict)]
        ggufs = [f for f in siblings if f.lower().endswith(".gguf")]
        if not ggufs:
            continue
        # Prefer explicit *-lora-*.gguf files; fall back to any .gguf in a LoRA repo
        lora_ggufs = [f for f in ggufs if "lora" in f.lower()] or (ggufs if "lora" in tags else [])
        if not lora_ggufs:
            continue
        base_model = _infer_base_from_repo(repo_id, tags)
        for fn in sorted(lora_ggufs):
            harness = (config.detect_harness(fn)
                       or config.detect_harness(repo_id)
                       or next((config.normalize_harness(t) for t in tags if config.normalize_harness(t) in config.KNOWN_HARNESSES), ""))
            adapter = config.detect_adapter(fn, repo_id, repo_id, harness)
            cached = cache / Path(fn).name
            out.append({
                "repo_id": repo_id,
                "filename": fn,
                "url": f"{HF_WEB}/{repo_id}/resolve/main/{urllib.parse.quote(fn)}",
                "base_model": base_model,
                "harness": harness,
                "adapter_name": adapter,
                "downloads": int(m.get("downloads") or 0),
                "likes": int(m.get("likes") or 0),
                "tags": tags,
                "cached_path": str(cached) if (cached.is_file() and cached.stat().st_size > 0) else "",
            })
    out.sort(key=lambda x: (-x["downloads"], x["base_model"].lower(), x["harness"], x["filename"]))
    return out


def fetch_gguf_adapters(author: str = DEFAULT_AUTHOR, base_filter: str = "", harness_filter: str = "",
                        timeout: float = 20.0) -> list[dict]:
    """Lists GGUF LoRA adapters published by `author` on Hugging Face."""
    params = {"author": author, "full": "true", "limit": "200"}
    if base_filter.strip():
        params["search"] = base_filter.strip()
    url = f"{HF_API}/models?{urllib.parse.urlencode(params)}"
    raw = _get_json(url, timeout=timeout)
    adapters = parse_gguf_models(raw if isinstance(raw, list) else [])
    if base_filter.strip():
        adapters = compatible_adapters(base_filter, adapters)
    if harness_filter.strip():
        want = config.normalize_harness(harness_filter)
        adapters = [a for a in adapters if a["harness"] == want or want in a["adapter_name"]]
    return adapters


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def compatible_adapters(base_model: str, adapters: list[dict]) -> list[dict]:
    """Filters adapters whose base model matches `base_model` (case/punctuation-insensitive)."""
    needle = _norm_key(config.split_model_quant(base_model)[0])
    if not needle:
        return list(adapters)
    matched = []
    for a in adapters:
        cand = _norm_key(a.get("base_model") or "")
        repo_k = _norm_key(a.get("repo_id") or "")
        if needle in cand or cand in needle or needle in repo_k:
            matched.append(a)
    return matched


def download_lora(repo_id: str, filename: str, on_line=None, timeout: float = 300.0) -> Path:
    """Downloads `filename` from `repo_id` into `~/.cache/gbench-tui/loras/` atomically. Skips if already cached."""
    say = on_line or (lambda _m: None)
    dest = loras_dir() / Path(filename).name
    if dest.is_file() and dest.stat().st_size > 0:
        say(f"✔ cached LoRA: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    url = f"{HF_WEB}/{repo_id}/resolve/main/{urllib.parse.quote(filename)}"
    say(f"downloading {repo_id}/{filename}…")
    headers = {"User-Agent": _UA}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, part.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            last_pct = -10
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total > 0:
                    pct = int((got * 100) / total)
                    if pct >= last_pct + 20:
                        last_pct = pct
                        say(f"  {pct}% ({got / 1e6:.1f} / {total / 1e6:.1f} MB)")
        part.replace(dest)
        say(f"✔ saved LoRA: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    except Exception:
        part.unlink(missing_ok=True)
        raise


def local_loras() -> list[Path]:
    """Scans ~/.cache/gbench-tui/loras and common folders for local LoRA GGUF files."""
    seen: dict[str, Path] = {}
    for d in (loras_dir(), Path.home(), Path.home() / "Downloads", Path.home() / "models"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.gguf")):
            if d == loras_dir() or "lora" in p.name.lower():
                seen[str(p.resolve())] = p
    return sorted(seen.values(), key=lambda p: p.name.lower())
