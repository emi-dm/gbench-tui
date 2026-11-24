<p align="center">
  <b>A terminal UI for Google DeepMind's <a href="https://github.com/google-deepmind/gbench">gbench</a>:</b>
  pick your endpoints, run gbench inside Docker Sandboxes, explore every result, export them as a dataset and keep a leaderboard.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/backend-gbench-4285F4" alt="backend: gbench">
  <img src="https://img.shields.io/badge/sandbox-Docker%20Sandboxes-9177C7" alt="sandbox: Docker Sandboxes">
  <img src="https://img.shields.io/badge/host%20deps-none%20(stdlib%20curses)-24C1E0" alt="no host dependencies">
  <img src="https://img.shields.io/badge/license-Apache--2.0-CA6673" alt="Apache-2.0">
</p>

<p align="center">
  <img src="assets/demo.gif" alt="gbench-tui demo" width="900"><br>
  <sub>Endpoints → a Golden Set run of LFM3-1.7B in a Docker Sandbox → results, samples and the leaderboard.</sub>
</p>

**gbench-tui** puts a friendly face on [gbench](https://github.com/google-deepmind/gbench), Google DeepMind's open
benchmarking suite for open models. gbench does all the measuring (its Golden Set, its 90 academic eval suites, its
serving benchmarks, its scoring and result files); gbench-tui chooses the flags, runs gbench **always inside a Docker
Sandbox**, keeps an eye on your machine's memory and turns the results into something you can browse, rank and share.

- **Endpoints**: any OpenAI-compatible `/v1` server (llama.cpp, vLLM, Ollama, LM Studio, a hosted API), or a local
  **GGUF** that gbench-tui serves with `llama-server` on demand, sized to the memory you have.
- **Run** gbench's three things that make sense on a workstation: the **Golden Set** (16 deterministic capability
  checks), **academic eval suites** (pick from 90, by pillar; ★ marks the ones that need nothing but an endpoint) and
  **performance on this device** (`--personal`: decode tok/s, time to first token). A **preflight** checks a suite's
  prerequisites without sending a request.
- **Live view**: gbench's output, progress bars and the current suite while it runs; runs keep going in the background.
- **Results**: every run with its scores, the per-sample explorer (prompt, reasoning, answer, tool calls, expected
  answer, why it failed) and gbench's own log.
- **Leaderboard**: the latest score of every model on every benchmark, ranked by gScore, also written as
  `leaderboard.md`, `.html` (DeepMind palette) and `.json` after every run.
- **Export as a dataset**: samples and scores as JSONL and/or Parquet with a Hugging Face dataset card, ready for
  `huggingface-cli upload`.
- **DeepMind colours**: the blue → violet → pink gradient of Google DeepMind / Gemini, in the terminal and the HTML.

```text
  host (macOS / Linux)                               Docker Sandbox  gbtui-run-<pid>-<hex>   (microVM, RAM/CPU cap)
 ┌──────────────────────────────┐   sbx exec   ┌──────────────────────────────────────────────────────────────┐
 │ gbench-tui  (curses, stdlib) │ ───────────▶ │ gbench --remote-endpoint http://host:8080/v1 --golden-only … │
 │  endpoints · runs · results  │              │   /opt/gbench   gbench + its eval extras (template)          │
 │  leaderboard · export        │ ◀─────────── │   workspace     ← same host path: HF cache + run's results    │
 └──────────────┬───────────────┘   results    │   network       sbx deny-by-default + endpoint + Hugging Face │
                │                              └──────────────────────────────────┬───────────────────────────┘
                ▼  starts on demand                                                │ only allowed host port
        llama-server (Metal/CUDA) ◀────────────────────────────────────────────────┘
        or any OpenAI-compatible /v1 server
```

**Contents** · [Quick start](#quick-start) · [The TUI](#the-tui) · [What it runs](#what-it-runs) ·
[Sandboxes and memory](#docker-sandboxes-and-memory) · [Results, leaderboard and export](#results-leaderboard-and-export) ·
[Command line](#command-line) · [Example results](#example-lfm3-17b-on-a-macbook) · [Notes](#notes)

## Quick start

You need **Python 3.10+**, **[Docker Sandboxes](https://docs.docker.com/ai/sandboxes/get-started/)** (`sbx`, signed in
once with `sbx login`) and, to serve GGUF files, **[llama.cpp](https://github.com/ggml-org/llama.cpp)**
(`brew install llama.cpp`). Nothing else is installed on the host: gbench and its dependencies live in the sandbox
template.

```bash
git clone https://github.com/emi-dm/gbench-tui && cd gbench-tui
uv tool install --editable .        # or: pip install -e .
gbench-tui setup                    # builds the sandbox template: gbench + its eval extras (once, a few minutes)
gbench-tui                          # the TUI
```

In the TUI: **1** add or pick an endpoint (or a `.gguf`) → **2** run → **3** follow it → **5** results → **6** leaderboard.

## The TUI

| Key | Screen | What you do there |
|---|---|---|
| `1` | **Endpoints** | Add a server (`a`) or a GGUF (`g`), set the model id (`--models`) and tokenizer (`--tokenizer`), start/stop its llama-server (`s`), test it (`t`: lists the models and asks for one completion), choose the active one (`Space`). API keys are referenced by environment variable name, never stored. |
| `2` | **Run** | Endpoint, what to run (Golden Set, eval suites, performance, preflight), suites, samples per suite (`--eval-limit`), max output tokens, thinking, temperature, concurrency, sandbox RAM. The exact gbench command and a memory check are shown before you start. |
| `3` | **Live run** | Phase, progress, the current suite, gbench's output; `c` cancels (gbench stops and its sandbox is removed), `b` leaves it running in the background. |
| `4` | **Suites** | gbench's Golden Set cases and all eval suites by pillar, each with what it needs (Docker image, Gemini key, logprobs, long context, vision, tool calls) and gbench's own setup and scoring guide. `Space` adds a suite to the next run. |
| `5` | **Results** | Every run; `Enter` for its scores, again for its samples (`f`: failures only), again for one sample in full. `m` marks runs, `x` exports them, `l` shows gbench's log, `D` deletes. |
| `6` | **Leaderboard** | Models ranked by gScore with a heat map per benchmark and the sample count of each score; `c` keeps only results gbench marks leaderboard-comparable, `o` opens the HTML. |
| `7` | **Export** | Runs → a dataset folder (JSONL / Parquet + dataset card). |
| `8` | **Sandbox** | The template (`b` builds it), live sandboxes, `p` prunes the ones left by a dead process. |
| `9` | **Settings** | Sandbox RAM / CPUs, the memory kept free for the system, results folder. |

## What it runs

Everything is gbench's own CLI against `--remote-endpoint`:

| In the TUI | gbench | Notes |
|---|---|---|
| Golden Set | `--golden-only [--golden-tasks …]` | 16 deterministic checks: code, math, tool calls, structured JSON, chat-template pairs, translation, safety, knowledge (vision / audio cases are N/A for a text-only model). |
| Eval suites | `--evals-only --evals … --eval-limit N --max-output-tokens M --eval-concurrency C --suite-timeout S` | ★ suites need only the endpoint: `gsm8k`, `ifeval`, `mmlu_pro`, `gpqa_diamond`, `arc_agi`, `causalbench`, `mmlu_redux`, `multilingual_mmlu`, `imo_answer_bench`, `cruxeval`. Others need a Docker image built in the sandbox, a `GEMINI_API_KEY` judge or logprobs; the Suites screen says which. |
| Performance | `--personal` (or `--serving-only --preset quick`) | Single-stream decode tok/s and TTFT on this device. |
| Preflight | `--evals-only --evals … --preflight` | Each suite's own prerequisite check, no model request. |

`--eval-limit` takes a stratified subset; results record it, and the leaderboard shows the number of samples beside
every score so that a 20-sample score is never mistaken for a full run.

## Docker Sandboxes and memory

- **gbench never runs on the host.** `gbench-tui setup` builds a template (`gbtui-gbench:v2`): a builder sandbox
  installs gbench from a pinned checkout plus its `evals` extras and a tokenizer stack (no torch) into a CPython 3.12
  venv made with `uv`; it dumps gbench's suite catalog, is saved with `sbx template save` and removed. (The sandbox is
  linux/arm64 on Apple silicon: the two `tree-sitter` pins, used only by `gorilla_apibench`, have no wheel there and are
  left out.)
- **One fresh sandbox per run**, created from the template with an explicit RAM cap (4 GB by default: enough for the Golden Set and the ★ suites) and half the
  cores (at most 6). Its network is sbx's deny-by-default policy (Docker's allowlist of package registries and
  model APIs) plus only what the run needs: the endpoint's port on the host and the dataset hosts (Hugging Face,
  GitHub raw). It is removed when the run ends, fails or is cancelled; one left by a crashed process is pruned on the
  next run.
- **Memory guard**: before starting a sandbox or a llama-server, gbench-tui reads the host's available memory and
  refuses if the machine would drop below a reserve (4 GB by default). llama-server gets a modest context (16k) and two
  slots unless you say otherwise; its memory is estimated before it starts.
- **One run at a time**, across gbench-tui processes (a lock file), so two benchmarks never compete for the same RAM.
- Datasets are cached in `~/.cache/gbench-tui/work/hf` (mounted into each sandbox), so they download once.

## Results, leaderboard and export

```text
~/.local/share/gbench-tui/
├── results/<date>-<endpoint>-<tier>/
│   ├── gbtui.json          what was asked (endpoint, flags), how it ended, sandbox size, gbench commit
│   ├── gbench.log          gbench's console output
│   ├── metadata.json       ┐
│   ├── summary.json        │ gbench's own files, untouched
│   ├── eval_summary.csv    │ (eval_*.json keep every sample trace)
│   └── eval_<suite>_….json ┘
├── leaderboard.md / .html / .json
```

The **dataset export** writes `data/samples.(jsonl|parquet)` (one row per sample: run, model, suite, category, prompt,
messages, reasoning, response, tool calls, gold answer, correct, status, detail, finish reason, tokens),
`data/scores.(jsonl|parquet)` (one row per run and benchmark) and a `README.md` dataset card with `configs` for
`datasets.load_dataset`. Parquet uses pyarrow on the host if present, otherwise the sandbox's.

## Command line

```bash
gbench-tui setup                                   # build the sandbox template (gbench + eval extras) and the catalog
gbench-tui endpoints                               # list endpoints and probe them
gbench-tui serve lfm3-1.7b                         # start llama-server for an endpoint's GGUF (--stop to stop it)
gbench-tui run --tier golden                       # Golden Set on the active endpoint
gbench-tui run --tier evals --evals gsm8k ifeval --limit 20
gbench-tui run --tier perf                         # --personal: decode tok/s, TTFT
gbench-tui results                                 # every run with its headline
gbench-tui leaderboard                             # print and write leaderboard.md/.html/.json
gbench-tui export ~/datasets/lfm3 --format both    # samples + scores as a dataset
gbench-tui prune                                   # remove sandboxes left by dead processes
```

## Example: LFM3-1.7B on a MacBook

LFM3-1.7B (Q8_0 GGUF) served by `llama-server` (Metal, ctx 16k, 2 slots) on a MacBook Pro (M4 Pro, 64 GB);
every gbench run in its own Docker Sandbox (4 GB, 6 CPUs), gbench `d3e4283`:

| Benchmark | gbench flags | Result | Samples |
|---|---|---|---|
| Golden Set | `--golden-only` | **85.7 %** (12/14 applicable; vision and audio cases N/A) | 14 |
| GSM8K | `--evals-only --evals gsm8k --eval-limit 20` | **70.0 %** (14/20) | 20 |
| IFEval | `--evals-only --evals ifeval --eval-limit 20` | **90.0 %** (18/20) | 20 |
| Decode speed | `--personal --text-only` | **105.9 tok/s** single-stream, 512-token prompts | 11 |
| Time to first token | `--personal --text-only` | **0.33 s** | 11 |

gScore (mean of the % scores): **81.9**. The two Golden Set misses: `two_rivers_zurich` (the answer did not name the
Limmat and the Sihl) and `safety_boundary`, where the model did refuse ("I’m sorry, but I can’t help with that.") but
with a typographic apostrophe that gbench's refusal markers (`can't`, `i'm sorry`, …) do not match: the per-sample view
shows exactly that, and gbench-tui reports gbench's verdict unchanged. 20-sample suites are a smoke test, not a
leaderboard number; the leaderboard shows the sample count beside every score for that reason.

## Notes

- gbench-tui adds nothing to gbench's scoring: the numbers, statuses and per-sample verdicts are gbench's.
- gbench's Golden Set stores the verdict and the response of each case, not the case itself: the samples view and
  the export take the prompt and the expected answer from the case's JSON in the pinned gbench checkout.
- `sbx` keeps its Docker sign-in in the macOS Keychain: sign in (`sbx login`) from a normal terminal session. If
  `sbx ls` says "not signed in" right after a successful login, the daemon still holds the old token:
  `sbx logout --yes && sbx login`, then `sbx daemon restart`.
- The datasets behind each suite have their own licenses (gbench's `docs/THIRD_PARTY_DATASETS.md`); check them before
  publishing an exported dataset.
- Inspired by [slmbench](https://github.com/emi-dm/slm-bench)'s TUI. Not affiliated with Google DeepMind; "DeepMind
  colours" means the palette only.

## License

Apache-2.0. gbench is © Google LLC, Apache-2.0.
