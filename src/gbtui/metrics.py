"""Metric definitions, exact calculation formulas, sources, and worked examples for gbench-tui.

Every formula documented here matches the actual implementation in `gbtui.results`, `gbtui.leaderboard`,
`gbtui.config`, `gbtui.servers`, and Google DeepMind's `gbench`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from . import catalog, config


@dataclass(frozen=True)
class MetricDef:
    id: str
    category: str
    name: str
    unit: str
    direction: str          # "higher" | "lower" | "dimension" | "guard"
    source: str             # where the value is computed
    formula: str            # concise mathematical formula (may be multi-line)
    summary: str            # 1-line description
    details: str            # full explanation of how it is computed
    example: str            # concrete worked calculation
    caveats: str = ""       # comparability / edge-case caveats


CATEGORIES: tuple[str, ...] = (
    "Leaderboard & Aggregation",
    "Accuracy & Capability",
    "Performance & Serving",
    "Quantization & Efficiency",
    "Web Charts & Host Safety",
)


METRICS: tuple[MetricDef, ...] = (
    # ---- Leaderboard & Aggregation ----
    MetricDef(
        id="gscore",
        category="Leaderboard & Aggregation",
        name="gScore",
        unit="% pts (0–100)",
        direction="higher",
        source="gbench-tui · leaderboard.build()",
        formula="gScore = (1 / K) × Σ score_i   [for all K benchmarks with unit = '%']",
        summary="Unweighted mean of a model variant's latest percentage benchmark scores",
        details=(
            "For each distinct (base_model, quant) variant, gbench-tui scans all runs in chronological "
            "order with status 'ok' or 'unknown' (failed and cancelled runs are ignored). Whenever a "
            "benchmark is re-run for the same variant, the newer score overwrites the older one. "
            "gScore is the arithmetic mean across all K percentage metrics (golden_set + eval suites + "
            "quality checks); throughput/latency metrics (tok/s, TTFT) are excluded from gScore."
        ),
        example="golden_set 85.7% + gsm8k 78.0% + ifeval 90.0% → (85.7 + 78.0 + 90.0) / 3 = 84.6",
        caveats=(
            "Strictly comparable across models only when evaluated on the same suites and --eval-limit. "
            "Press 'c' in Leaderboard to filter evals to gbench's leaderboard_comparable=true runs only."
        ),
    ),
    MetricDef(
        id="delta_rank",
        category="Leaderboard & Aggregation",
        name="Δ vs #1 & Δ vs Quant",
        unit="± pts",
        direction="higher",
        source="gbench-tui · Leaderboard & By Quant chart",
        formula="Δ_leader = gScore_variant − gScore_#1\nΔ_quant  = gScore_quant − gScore_highest_quant  (same base_model)",
        summary="Score gap against the #1 leader or against the highest-precision quantization",
        details=(
            "In the TUI Leaderboard preview, Δ vs #1 shows how many gScore points separate the selected "
            "variant from the top-ranked entry. In the web 'By Quant' bar chart, Δ vs ref compares each "
            "quantization of a base model against its highest-bit evaluated quantization (e.g. Q4_K_M vs "
            "Q8_0 or BF16) to quantify accuracy retention under compression."
        ),
        example="LFM3-1.7B Q4_K_M (76.0) vs Q8_0 (84.6) → Δ = -8.6 pts vs Q8_0",
        caveats="Δ vs Quant requires at least two evaluated quantizations of the same base model.",
    ),
    # ---- Accuracy & Capability ----
    MetricDef(
        id="golden_set",
        category="Accuracy & Capability",
        name="golden_set (Golden Set)",
        unit="% (0–100)",
        direction="higher",
        source="gbench --golden-only · summary.json",
        formula="accuracy_percent = (passed_cases / applicable_tasks) × 100",
        summary="16 deterministic capability smoke tests across core LLM behaviors",
        details=(
            "Evaluates 16 deterministic checks: math, code generation, tool calling, structured JSON, "
            "chat-template fidelity, translation, factual knowledge, and safety boundaries. Cases whose "
            "modality is unsupported by the endpoint (e.g. vision/audio tasks on a text-only model) are "
            "flagged 'not_applicable' and excluded from applicable_tasks so text-only models are not penalized."
        ),
        example="12 passed, 2 failed, 2 N/A → applicable_tasks = 14 → 12 / 14 × 100 = 85.7%",
        caveats=(
            "gbench exits with rc=1 if any golden check fails (still recorded as a valid completed run). "
            "Verifiers are strict (exact schema/markers); inspect Results → Samples for per-case reasons."
        ),
    ),
    MetricDef(
        id="eval_accuracy",
        category="Accuracy & Capability",
        name="Eval Suite Accuracy",
        unit="% (0–100)",
        direction="higher",
        source="gbench --evals-only · eval_<suite>_*.json",
        formula="accuracy = (correct_answers / total_questions) × 100  [or suite-native metric]",
        summary="Academic benchmark accuracy scored by each suite's official gbench verifier",
        details=(
            "gbench-tui never re-scores outputs: it displays the exact accuracy and per-sample verdicts "
            "reported by gbench. Verifiers vary by suite (numeric extraction for GSM8K, programmatic "
            "instruction checks for IFEval, multiple-choice for MMLU-Pro/GPQA, sandboxed code execution "
            "for coding suites, or Gemini LLM-as-judge when required). Each cell shows n = total_questions."
        ),
        example="gsm8k with --eval-limit 20: 15 correct / 20 questions = 75.0% (n=20)",
        caveats=(
            "Truncating at --max-output-tokens counts as a failure. Using --eval-limit N evaluates a "
            "stratified subset and sets leaderboard_comparable = false."
        ),
    ),
    MetricDef(
        id="sample_pass_rate",
        category="Accuracy & Capability",
        name="Sample Pass Rate",
        unit="% (0–100)",
        direction="higher",
        source="gbench-tui · Results → Samples view",
        formula="pass_rate = passed / (passed + failed) × 100",
        summary="Ratio of passed to graded samples in the per-sample inspector",
        details=(
            "Shown above the stacked green/red/grey bar in the Samples explorer. Samples marked N/A "
            "(correct = None) are displayed separately and omitted from the pass_rate denominator so the "
            "bar matches gbench's applicable-task accuracy."
        ),
        example="12 passed, 2 failed, 2 N/A → 12 / (12 + 2) × 100 = 85.7%",
        caveats="Reflects all samples in the selected score (unaffected by the active all/fail/pass filter).",
    ),
    # ---- Performance & Serving ----
    MetricDef(
        id="decode_tok_s",
        category="Performance & Serving",
        name="decode tok/s",
        unit="tok/s",
        direction="higher",
        source="gbench --personal / --serving-only · results._perf_scores()",
        formula="decode tok/s = 1000.0 / TPOT_ms",
        summary="Steady-state single-stream token generation speed after the first token",
        details=(
            "Derived from gbench's Time Per Output Token (TPOT) in milliseconds using priority order: "
            "tpot_p50_ms → median_tpot_ms → median_tpot_ms_mean → mean_tpot_ms → mean_tpot_ms_mean. "
            "Inverting TPOT_ms gives the sustained decode rate in tokens/second, tagged by campaign or "
            "prompt input length (e.g. 'decode tok/s (in512)')."
        ),
        example="median_tpot_ms = 9.44 ms → 1000.0 / 9.44 = 105.93 tok/s",
        caveats="Single-stream speed on local hardware depends heavily on memory bandwidth and quantization bits.",
    ),
    MetricDef(
        id="ttft",
        category="Performance & Serving",
        name="TTFT (Time To First Token)",
        unit="seconds (s)",
        direction="lower",
        source="gbench --personal / --serving-only · results._perf_scores()",
        formula="TTFT_s = TTFT_ms / 1000.0",
        summary="Latency from sending the prompt until the first output token arrives",
        details=(
            "Taken from gbench's ttft_p50_ms → median_ttft_ms → mean_ttft_ms and converted from "
            "milliseconds to seconds (rounded to 3 decimals). Captures prompt prefill compute time plus "
            "any server queueing overhead for the tagged prompt length (e.g. in512)."
        ),
        example="median_ttft_ms = 330 ms → 330 / 1000.0 = 0.330 s",
        caveats="Lower is better; on Radar/Polar charts TTFT is inverted so faster TTFT extends further out.",
    ),
    MetricDef(
        id="output_tok_s",
        category="Performance & Serving",
        name="output tok/s",
        unit="tok/s",
        direction="higher",
        source="gbench --serving-only · results._perf_scores()",
        formula="output tok/s = output_tokens_per_second_mean  (or output_throughput)",
        summary="Aggregate output throughput across concurrent serving requests",
        details=(
            "Directly reads gbench's aggregate output throughput reported during multi-request serving "
            "benchmarks (total generated tokens across all active slots divided by total elapsed time)."
        ),
        example="2 concurrent slots generating 85 tok/s each → ~170.0 tok/s aggregate output",
        caveats="Present only when gbench reports serving throughput (typically --serving-only).",
    ),
    # ---- Quantization & Efficiency ----
    MetricDef(
        id="quant_dimension",
        category="Quantization & Efficiency",
        name="Quantization & bpw",
        unit="tag · bpw",
        direction="dimension",
        source="gbench-tui · config.detect_quant() & quant_bits()",
        formula="variant_key = (base_model.lower(), canonical_quant)\nbpw         = effective bits-per-weight lookup (e.g. Q4_K_M ≈ 4.85, Q8_0 ≈ 8.50)",
        summary="First-class model variant dimension separating quantizations of the same base model",
        details=(
            "Quantization is resolved in order: explicit Endpoint.quant → GGUF filename → served model id "
            "→ endpoint name. Recognizes GGUF/llama.cpp (Q2_K..Q8_0, IQ1_S..IQ4_XS, UD-Q4_K_XL), floating "
            "precision (F32, F16, BF16, FP8, MXFP4), and format tags (AWQ, GPTQ, 4BIT, 8BIT). Because the "
            "leaderboard keys by (base_model, quant), LFM3-1.7B Q4_K_M and Q8_0 never overwrite each other."
        ),
        example="LFM3-1.7B_Q4_K_M.gguf → base='LFM3-1.7B', quant='Q4_K_M', bpw=4.85",
        caveats="You can override auto-detection anytime via the Quantization field in 1 · Endpoints.",
    ),
    MetricDef(
        id="quant_compactness",
        category="Quantization & Efficiency",
        name="compact (bits)",
        unit="% (10–100)",
        direction="higher",
        source="gbench-tui · web leaderboard Radar / Polar chart",
        formula="compact = clamp(10, 100, 100 − ((log₂(bpw) − 1) / 4) × 90)",
        summary="Log-scaled memory compactness score for the circular area charts",
        details=(
            "Maps effective bits-per-weight (bpw) onto a 10–100 radial scale so smaller quantizations "
            "extend further out on the 'compact (bits)' axis when comparing accuracy vs. footprint on "
            "the web Radar and Polar Area charts. Each halving of bit-width adds +22.5 points."
        ),
        example="Q2_K (2.63) → 91% · Q4_K_M (4.85) → 71% · Q6_K (6.56) → 61% · Q8_0 (8.50) → 53% · BF16 (16) → 33%",
        caveats="Enabled via the 'quant' checkbox in leaderboard.html; does not affect gScore.",
    ),
    # ---- Web Charts & Host Safety ----
    MetricDef(
        id="radar_norm",
        category="Web Charts & Host Safety",
        name="Radar / Polar Normalization",
        unit="0–100 radius",
        direction="higher",
        source="gbench-tui · leaderboard.html buildDimensions()",
        formula=(
            "pct (%)    : norm = value\n"
            "tok/s (↑)  : norm = min(100, (value / max_value) × 100)\n"
            "TTFT  (↓)  : norm = min(100, (min_value / value) × 100)\n"
            "quant (↓b) : norm = compact(bpw)"
        ),
        summary="How heterogeneous metrics are scaled onto the 0–100 circular area charts",
        details=(
            "Percentage benchmarks already lie on 0–100 and are plotted directly. When 'perf' is toggled "
            "on, throughput metrics scale linearly relative to the fastest evaluated model (=100) while "
            "TTFT scales inversely relative to the lowest-latency model (=100). Hover tooltips always show "
            "the exact raw unnormalized numbers."
        ),
        example="Model A TTFT 0.14s (best) → 100; Model B TTFT 0.28s → (0.14 / 0.28) × 100 = 50",
        caveats="Relative perf normalization updates dynamically as new faster models enter the leaderboard.",
    ),
    MetricDef(
        id="ram_estimate",
        category="Web Charts & Host Safety",
        name="Host Memory Guard & RAM Est.",
        unit="bytes / GB",
        direction="guard",
        source="gbench-tui · servers.estimate_bytes() & sysmem.check()",
        formula="est_bytes = gguf_size × 1.05 + ctx × 0.06 MB + 600 MB\nallow     = (host_available_RAM − est_bytes) ≥ min_free_gb",
        summary="Pre-flight host RAM estimate that prevents llama-server and sandboxes from swapping",
        details=(
            "Before launching llama-server for a local GGUF or starting a Docker container, "
            "gbench-tui reads current available host memory and compares it against the estimated weight "
            "+ KV-cache + buffer footprint (plus sandbox RAM cap). Launch is blocked if free RAM would "
            "fall below Settings.min_free_gb (default 4.0 GB)."
        ),
        example="1.8 GB GGUF + ctx 16384 → 1.89 GB + 0.98 GB + 0.60 GB ≈ 3.47 GB estimated",
        caveats="KV-cache coefficient (~0.06 MB/token) is an upper-bound heuristic tailored for 1–4B GQA models.",
    ),
)


def by_category() -> list[tuple[str, list[MetricDef]]]:
    groups: dict[str, list[MetricDef]] = {c: [] for c in CATEGORIES}
    for m in METRICS:
        groups.setdefault(m.category, []).append(m)
    return [(c, ms) for c, ms in groups.items() if ms]


def quant_reference_rows() -> list[tuple[str, float]]:
    """Canonical representative quantizations sorted from smallest to full precision."""
    sample = ("IQ2_XS", "Q2_K", "Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "FP8", "Q8_0", "BF16", "F32")
    return [(q, config.quant_bits(q) or 0.0) for q in sample]


# ------------------------------------------------------------------------------------------------
# Per-benchmark metrics (parsed from gbench's own docs/evals/<suite>.md + curated fallbacks)
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkMetric:
    name: str
    title: str
    pillar: str
    light: bool
    metric_tag: str              # compact metric label shown in lists & badges
    headline: str                # how the primary metric is calculated/scored
    protocol: str                # prompt / shots / repeats / verifier protocol
    sub_metrics: tuple[str, ...] # companion / breakdown metrics reported in JSON
    needs: tuple[str, ...] = ()
    has_doc: bool = False


# Curated metric tags & offline fallbacks (used when gbench-src docs aren't cached yet, and for crisp tags).
_BENCH_CURATED: dict[str, dict] = {
    "golden_set": {
        "title": "Golden Set (16 checks)",
        "pillar": "Golden Set",
        "light": True,
        "metric_tag": "applicable pass %",
        "headline": "accuracy_percent = (passed_cases / applicable_tasks) × 100. Cases whose modality is unsupported (vision/audio on text-only models) are marked not_applicable and omitted from applicable_tasks.",
        "protocol": "16 deterministic smoke tests covering math, code, tool calling, structured JSON, chat-template pairs, translation, safety refusals, and factual knowledge.",
        "sub_metrics": ("passed_cases", "failed_cases", "error_cases", "not_applicable_cases"),
    },
    "personal_perf": {
        "title": "Performance (--personal)",
        "pillar": "Performance",
        "light": True,
        "metric_tag": "tok/s + TTFT",
        "headline": "decode tok/s = 1000.0 / TPOT_ms (higher is better) and TTFT_s = TTFT_ms / 1000.0 (lower is better), using median/p50 latencies per prompt-length campaign (e.g. in512). --serving-only also reports aggregate output tok/s.",
        "protocol": "Single-stream interactive timing on the target device (--personal) measuring prefill time-to-first-token (TTFT) and inter-token decode latency (TPOT).",
        "sub_metrics": ("decode tok/s", "TTFT", "output tok/s", "completed_requests"),
    },
    "gsm8k": {
        "title": "GSM8K",
        "pillar": "Math & Logic",
        "light": True,
        "metric_tag": "strict exact_match",
        "headline": "accuracy is exact_match (%) under the strict-match filter on the numeric answer after 'The answer is'. Also records flexible_extract_accuracy.",
        "protocol": "lm-evaluation-harness gsm8k_cot: 1,319 grade-school arithmetic problems, 8-shot chain-of-thought exemplars, greedy decoding.",
        "sub_metrics": ("strict-match", "flexible_extract_accuracy", "replies_over_max_gen_toks"),
    },
    "ifeval": {
        "title": "IFEval",
        "pillar": "Instruction Following",
        "light": True,
        "metric_tag": "prompt-strict %",
        "headline": "accuracy is prompt-level strict accuracy (%): a prompt passes only when the response satisfies every verifiable instruction constraint in its list.",
        "protocol": "541 prompts with programmatic verifiable constraints (word count, JSON format, forbidden words, casing, etc.) using Google Research's official checker.",
        "sub_metrics": ("prompt_level_strict", "instruction_level_strict", "prompt_level_loose", "instruction_level_loose"),
    },
    "causalbench": {
        "title": "CausalBench (Corr2Cause)",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": True,
        "metric_tag": "F1 (valid class)",
        "headline": "accuracy is the F1 score (%) of the minority (valid) class — plain accuracy is misleading because ~85% of items are invalid ('No'). Also reports raw_accuracy, precision, and recall.",
        "protocol": "1,162 Corr2Cause causal-inference items testing whether a causal hypothesis follows from stated statistical correlations.",
        "sub_metrics": ("f1", "precision", "recall", "raw_accuracy"),
    },
    "gpqa_diamond": {
        "title": "GPQA Diamond",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": True,
        "metric_tag": "mean acc (10×)",
        "headline": "accuracy is the mean multiple-choice accuracy (%) across 10 shuffled-option repeats of the 198 expert questions (simple-evals protocol).",
        "protocol": "198 PhD-level physics, chemistry, and biology 4-option multiple-choice questions repeated 10 times with randomized choice order.",
        "sub_metrics": ("per_repeat_accuracy", "category_accuracy", "n_repeats"),
    },
    "mmlu_pro": {
        "title": "MMLU-Pro",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": True,
        "metric_tag": "exact match (A–J)",
        "headline": "accuracy is the percentage of test questions whose extracted answer letter (A–J) matches the gold letter, pooled over all scored questions across 14 domains.",
        "protocol": "12,032 reasoning-focused 10-option multiple-choice questions run via TIGER-Lab's chain-of-thought prompt protocol.",
        "sub_metrics": ("category_accuracy", "correct_answers", "total_questions"),
    },
    "mmlu_redux": {
        "title": "MMLU-Redux",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": True,
        "metric_tag": "exact match %",
        "headline": "accuracy is exact match (%) pooled over all error-audited MMLU rows across subjects.",
        "protocol": "Human-reannotated subset of MMLU removing ground-truth errors, scored by exact choice extraction.",
        "sub_metrics": ("category_accuracy", "correct_answers"),
    },
    "multilingual_mmlu": {
        "title": "Multilingual MMLU",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": True,
        "metric_tag": "pooled acc %",
        "headline": "accuracy is the percentage of questions answered correctly, pooled across all evaluated languages.",
        "protocol": "Translated MMLU multiple-choice questions scored per language and pooled overall.",
        "sub_metrics": ("category_accuracy", "per_language_accuracy"),
    },
    "arc_agi": {
        "title": "ARC-AGI",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": True,
        "metric_tag": "pass@2 (per task)",
        "headline": "accuracy is pass@2 (%) averaged per task: a test input-output grid pair is solved if either of two attempts matches the exact output grid.",
        "protocol": "ARC-AGI-1 public evaluation set: infer abstract grid transformations from demonstration pairs and emit exact 2D integer grids.",
        "sub_metrics": ("pass_at_2", "pass_at_1", "solved_tasks"),
    },
    "cruxeval": {
        "title": "CRUXEval",
        "pillar": "Coding",
        "light": True,
        "metric_tag": "pass@1 (I & O)",
        "headline": "accuracy is the mean of CRUXEval-I (input prediction) pass@1 and CRUXEval-O (output prediction) pass@1 using the unbiased pass@k estimator.",
        "protocol": "800 short Python functions evaluated on both predicting output from input (O) and finding a valid input for a target output (I).",
        "sub_metrics": ("cruxeval_i_pass_at_1", "cruxeval_o_pass_at_1"),
    },
    "imo_answer_bench": {
        "title": "IMO-AnswerBench",
        "pillar": "Math & Logic",
        "light": True,
        "metric_tag": "mean acc (≥8×)",
        "headline": "accuracy = correct attempts / graded attempts averaged across multiple runs (8+ passes) of competition math problems.",
        "protocol": "Olympiad-level short-answer mathematics problems graded by symbolic/exact equivalence across repeated attempts.",
        "sub_metrics": ("correct_attempts", "graded_attempts"),
    },
    "bigcodebench": {
        "title": "BigCodeBench",
        "pillar": "Coding",
        "light": False,
        "metric_tag": "calibrated pass@1",
        "headline": "accuracy is calibrated pass@1 (%) on the Instruct split; gbench also evaluates the Complete split using real unittest TestCases in Docker.",
        "protocol": "Practical Python library tasks (pandas, numpy, matplotlib, sklearn, requests) executed against hidden unittest suites.",
        "sub_metrics": ("bigcodebench_pass_at_1", "instruct", "complete"),
    },
    "bfcl": {
        "title": "BFCL",
        "pillar": "Tool Use & Agents",
        "light": False,
        "metric_tag": "BFCL Overall Acc",
        "headline": "accuracy is BFCL Overall Acc (%), combining AST/executable single-turn, relevance/irrelevance, and multi-turn function-calling scores.",
        "protocol": "Berkeley Function Calling Leaderboard categories graded by AST schema checks and state-based execution verifiers.",
        "sub_metrics": ("non_live", "live", "multi_turn", "relevance"),
    },
    "cimemories": {
        "title": "CIMemories",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": False,
        "metric_tag": "Complete + Viol@5",
        "headline": "accuracy reports Completeness (%), paired with violation_at_5 (%, lower is better) measuring private-attribute leakage across 5 generations via REVEAL judge.",
        "protocol": "Contextual-integrity privacy benchmark over synthetic user memories across 3 Westin privacy personas.",
        "sub_metrics": ("completeness", "violation_at_5"),
    },
    "docvqa": {
        "title": "DocVQA",
        "pillar": "Multimodal",
        "light": False,
        "metric_tag": "ANLS × 100",
        "headline": "accuracy is mean ANLS (Average Normalized Levenshtein Similarity) × 100 across document visual-QA questions.",
        "protocol": "Document image reading comprehension scored with edit-distance similarity thresholded at 0.5.",
        "sub_metrics": ("anls",),
    },
    "chartqa": {
        "title": "ChartQA",
        "pillar": "Multimodal",
        "light": False,
        "metric_tag": "relaxed acc (5%)",
        "headline": "accuracy is relaxed accuracy (%) over 2,500 chart questions (numeric answers within 5% relative tolerance count as correct).",
        "protocol": "Human and augmented visual chart reasoning questions over plotted figures.",
        "sub_metrics": ("human", "augmented"),
    },
    "swe_bench_pro": {
        "title": "SWE-bench Pro",
        "pillar": "Coding",
        "light": False,
        "metric_tag": "resolved rate %",
        "headline": "accuracy is the re-graded resolved rate (%) over scored repository issues whose patches pass all hidden fail_to_pass and pass_to_pass tests.",
        "protocol": "642 multi-language (Go/Python/JS/TS) real repository PR tasks executed via Harbor Docker sandboxes.",
        "sub_metrics": ("in_sandbox", "category_accuracy", "scored_instances"),
    },
    "healthbench": {
        "title": "HealthBench",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": False,
        "metric_tag": "rubric score %",
        "headline": "accuracy = clip(mean of per-conversation physician rubric scores, 0, 1) × 100, graded by an LLM judge against weighted clinical criteria.",
        "protocol": "5,000 multi-turn health conversations scored against physician-authored positive and negative rubric items.",
        "sub_metrics": ("rubric_score", "category_accuracy"),
    },
    "simpleqa": {
        "title": "SimpleQA",
        "pillar": "General Knowledge & Scientific Reasoning",
        "light": False,
        "metric_tag": "LLM judge acc %",
        "headline": "accuracy is simple-evals is_correct (%) over graded short-answer factual questions, plus not-attempted / incorrect calibration breakdown.",
        "protocol": "Factuality benchmark graded as CORRECT / INCORRECT / NOT_ATTEMPTED by a judge model.",
        "sub_metrics": ("is_correct", "is_incorrect", "is_not_attempted", "f_score"),
    },
    "aime": {
        "title": "AIME",
        "pillar": "Math & Logic",
        "light": False,
        "metric_tag": "mean acc (avg@N)",
        "headline": "accuracy is mean accuracy (%) averaged over repeated competition runs (MathArena protocol), verifying integer answers 0–999.",
        "protocol": "AIME competition math problems evaluated across repeated sampling passes with symbolic/numeric extraction.",
        "sub_metrics": ("per_run_accuracy", "pass_at_k"),
    },
}

_METRIC_TAG_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bviolation_at_5\b|\bviolation@5\b", "Complete + Viol@5"),
    (r"\banls\b", "ANLS × 100"),
    (r"\brelaxed accuracy\b", "relaxed acc (5%)"),
    (r"\bpass@2\b", "pass@2 %"),
    (r"\bcalibrated pass@1\b", "calibrated pass@1"),
    (r"\bpass@1\b", "pass@1 %"),
    (r"\bpass@k\b", "pass@k %"),
    (r"\bf1\b", "F1 score %"),
    (r"\bwin[- ]?rate\b|wins-or-ties", "win rate %"),
    (r"\bresolved rate\b|\bresolved\b", "resolved rate %"),
    (r"\bprompt-level strict\b", "prompt-strict %"),
    (r"\bstrict-match\b", "exact_match (strict)"),
    (r"\bexact[ _]match\b", "exact match %"),
    (r"\brubric\b", "rubric score %"),
    (r"\bcider\b|\bbleu\b|\brouge\b", "text-sim score"),
    (r"\blogprob", "logprob acc %"),
    (r"\bllm[- ]?(?:as[- ]?a[- ]?)?judge\b|\bgrader\b", "LLM judge %"),
    (r"\bstate verifier", "state verifier %"),
    (r"\bmean (?:score|accuracy)\b", "mean accuracy %"),
)

_SUB_NOISE = {
    "accuracy", "gbench", "json", "csv", "stdout", "stderr", "true", "false", "null", "none", "test", "train",
    "validation", "main", "all", "default", "standard", "docker", "python", "eval_summary.csv", "summary.json",
    "metadata.json", "prompt", "response", "messages", "model", "models", "evals", "eval", "hard", "easy",
}


def _clean_md(text: str) -> str:
    s = re.sub(r"```.*?```", " ", text, flags=re.S)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"^\s*[-*>]\s*", "", s, flags=re.M)
    return " ".join(s.split()).strip()


def _trim_prose(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Prefer ending at the last sentence boundary if it keeps enough text
    for sep in (". ", "; "):
        idx = cut.rfind(sep)
        if idx >= limit // 2:
            return cut[: idx + 1].rstrip()
    return cut.rsplit(" ", 1)[0].rstrip(",:;- ") + "…"


def _infer_metric_tag(text: str, needs: list[str] | tuple[str, ...]) -> str:
    low = text.lower()
    for pat, tag in _METRIC_TAG_PATTERNS:
        if re.search(pat, low):
            return tag
    if "Gemini key" in needs:
        return "LLM judge %"
    if "logprobs" in needs:
        return "logprob acc %"
    if "docker image" in needs:
        return "exec pass %"
    return "accuracy %"


def _extract_sub_metrics(raw_sec: str) -> tuple[str, ...]:
    found = []
    for tok in re.findall(r"`([a-z][a-z0-9_.-]{2,34})`", raw_sec):
        low = tok.lower()
        if low in _SUB_NOISE or "/" in tok or tok.endswith((".py", ".md", ".json", ".csv", ".txt", ".yaml")):
            continue
        if low.startswith(("gbench_", "vllm_", "hf_", "openai_", "gemini_")):
            continue
        if any(k in low for k in ("acc", "pass", "f1", "prec", "rec", "rate", "score", "viol", "comp", "strict",
                                  "loose", "repeat", "category", "anls", "bleu", "rouge", "cider", "win",
                                  "sandbox", "split", "exact", "flex", "turn", "tool", "error", "correct", "metric")):
            if tok not in found:
                found.append(tok)
    return tuple(found[:6])


def _parse_suite_doc(name: str, suite_meta: dict) -> BenchmarkMetric:
    cur = _BENCH_CURATED.get(name, {})
    title = suite_meta.get("title") or cur.get("title") or name
    pillar = suite_meta.get("pillar") or cur.get("pillar") or "Other"
    light = bool(suite_meta.get("light", cur.get("light", False)))
    needs_t = tuple(suite_meta.get("needs") or ())

    doc_path = catalog.src_dir() / "docs" / "evals" / f"{name}.md"
    raw = ""
    if doc_path.is_file():
        try:
            raw = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw = ""

    if not raw:
        headline = cur.get("headline") or suite_meta.get("description") or "Reported directly by gbench as accuracy (%)."
        protocol = cur.get("protocol") or suite_meta.get("description") or "Standard gbench evaluation protocol."
        tag = cur.get("metric_tag") or _infer_metric_tag(headline + " " + protocol, needs_t)
        return BenchmarkMetric(
            name=name, title=title, pillar=pillar, light=light, metric_tag=tag,
            headline=headline, protocol=protocol, sub_metrics=tuple(cur.get("sub_metrics", ())),
            needs=needs_t, has_doc=False,
        )

    secs: dict[str, str] = {}
    for m in re.finditer(r"(?ms)^##\s+([^\n]+)\n(.*?)(?=^##\s|\Z)", raw):
        secs[m.group(1).strip()] = m.group(2).strip()
    m0 = re.match(r"^#[^\n]+\n+(.+?)(?=\n## |\Z)", raw, re.S)
    intro = m0.group(1).strip() if m0 else ""

    raw_metric = ""
    for pat in ("headline", "reading the result", "scoring", "how it is scored", "what is measured", "metric"):
        for k, v in secs.items():
            if pat in k.lower():
                raw_metric = v
                break
        if raw_metric:
            break
    raw_proto = ""
    for pat in ("protocol", "what it measures", "how gbench runs", "overview", "context window"):
        for k, v in secs.items():
            if pat in k.lower():
                raw_proto = v
                break
        if raw_proto:
            break

    headline = _trim_prose(_clean_md(raw_metric), 620) if raw_metric else (cur.get("headline") or _trim_prose(_clean_md(intro), 500))
    protocol = _trim_prose(_clean_md(raw_proto), 560) if raw_proto else (cur.get("protocol") or _trim_prose(_clean_md(intro), 500))
    if not headline:
        headline = suite_meta.get("description") or "Reported directly by gbench as accuracy (%)."
    if not protocol:
        protocol = suite_meta.get("description") or "Standard gbench evaluation protocol."
    tag = cur.get("metric_tag") or _infer_metric_tag(headline + " " + _clean_md(intro), needs_t)
    subs = _extract_sub_metrics(raw_metric or raw[:2000]) or tuple(cur.get("sub_metrics", ()))
    return BenchmarkMetric(
        name=name, title=title, pillar=pillar, light=light, metric_tag=tag,
        headline=headline, protocol=protocol, sub_metrics=subs, needs=needs_t, has_doc=True,
    )


@lru_cache(maxsize=1)
def benchmark_metrics() -> list[BenchmarkMetric]:
    """All benchmarks with their scoring metrics: Golden Set, Performance, plus every gbench eval suite."""
    out: list[BenchmarkMetric] = []
    for special in ("golden_set", "personal_perf"):
        c = _BENCH_CURATED[special]
        out.append(BenchmarkMetric(
            name=special, title=c["title"], pillar=c["pillar"], light=True,
            metric_tag=c["metric_tag"], headline=c["headline"], protocol=c["protocol"],
            sub_metrics=tuple(c["sub_metrics"]), needs=(), has_doc=False,
        ))
    cat = catalog.load()
    suites = cat.get("suites") or []
    seen = set()
    if suites:
        pillars_order = list(cat.get("pillars") or []) + ["Other"]
        for p in pillars_order:
            group = [s for s in suites if s.get("pillar") == p]
            for s in sorted(group, key=lambda x: (not x.get("light"), x["name"])):
                if s["name"] in seen:
                    continue
                seen.add(s["name"])
                out.append(_parse_suite_doc(s["name"], s))
    else:
        # Offline fallback before gbench-tui setup has populated catalog.json / gbench-src
        for name, c in _BENCH_CURATED.items():
            if name in ("golden_set", "personal_perf"):
                continue
            out.append(_parse_suite_doc(name, c))
    return out


def benchmark_metric(name: str) -> BenchmarkMetric | None:
    key = str(name or "").strip().lower()
    if not key:
        return None
    if key in ("golden", "golden_set"):
        key = "golden_set"
    elif key.startswith(("decode", "ttft", "output", "perf", "personal")):
        key = "personal_perf"
    for b in benchmark_metrics():
        if b.name.lower() == key or b.title.lower() == key:
            return b
    return None


def reload() -> None:
    benchmark_metrics.cache_clear()


def format_benchmark_plain(bm: BenchmarkMetric) -> str:
    lines = [
        f"Benchmark · {bm.name} ({bm.title})",
        "=" * (12 + len(bm.name) + len(bm.title) + 3),
        f"Pillar        : {bm.pillar}" + ("  [★ laptop-friendly]" if bm.light else ""),
        f"Primary metric: {bm.metric_tag}",
        f"How scored    : {bm.headline}",
        f"Protocol      : {bm.protocol}",
    ]
    if bm.sub_metrics:
        lines.append(f"Sub-metrics   : {', '.join(bm.sub_metrics)}")
    if bm.needs:
        lines.append(f"Needs         : {', '.join(bm.needs)}")
    return "\n".join(lines) + "\n"


def format_plain(bench_name: str | None = None) -> str:
    """Plain-text summary of core formulas or a specific benchmark's metrics (for `gbench-tui metrics`)."""
    if bench_name:
        bm = benchmark_metric(bench_name)
        if bm:
            return format_benchmark_plain(bm)
        # Fallback: substring match across core metrics
        q = bench_name.lower()
        hits = [m for m in METRICS if q in m.id.lower() or q in m.name.lower()]
        if hits:
            lines = []
            for m in hits:
                lines += [f"{m.name} ({m.unit})", f"  formula: {m.formula}", f"  details: {m.details}",
                          f"  example: {m.example}", ""]
            return "\n".join(lines).rstrip() + "\n"
        return f"No metric or benchmark matching {bench_name!r}.\n"
    lines = ["gbench-tui · Metrics & Formulas", "=" * 32, ""]
    for cat, items in by_category():
        lines.append(f"[{cat}]")
        for m in items:
            arrow = {"higher": "↑", "lower": "↓", "dimension": "◈", "guard": "◆"}.get(m.direction, "·")
            lines.append(f"  {arrow} {m.name}  ({m.unit})")
            for fl in m.formula.splitlines():
                lines.append(f"      formula : {fl}")
            lines.append(f"      source  : {m.source}")
            lines.append(f"      example : {m.example}")
            if m.caveats:
                lines.append(f"      note    : {m.caveats}")
            lines.append("")
    bms = benchmark_metrics()
    lines.append(f"[Per-Benchmark Metrics ({len(bms)} available — run `gbench-tui metrics <suite>` for full details)]")
    for b in bms[:16]:
        star = "★" if b.light else " "
        lines.append(f"  {star} {b.name:<22} {b.metric_tag:<22} {b.pillar}")
    if len(bms) > 16:
        lines.append(f"  … and {len(bms) - 16} more suites (browse all in TUI: 0 · Metrics → By Benchmark)")
    return "\n".join(lines).rstrip() + "\n"
