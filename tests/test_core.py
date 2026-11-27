"""Unit tests that need neither sbx nor a model: gbench's command line, result parsing, leaderboard and export."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["GBTUI_HOME"] = tempfile.mkdtemp()
os.environ["GBTUI_CACHE"] = tempfile.mkdtemp()

from gbtui import config, export, leaderboard, results as R, runner, sandbox  # noqa: E402


def ep(**kw):
    return config.Endpoint(name="lfm3", base_url="http://127.0.0.1:8080/v1", model="lfm3-1.7b",
                           tokenizer="LiquidAI/LFM3-1.7B", **kw)


def fake_run(root: Path, rid: str, model: str, golden=None, evals=(), perf=None, status="ok"):
    d = root / rid
    d.mkdir(parents=True)
    models = []
    if golden is not None:
        models.append({"benchmark_type": "golden", "accuracy_percent": golden, "applicable_tasks": 10, "passed_cases": 7,
                       "failed_cases": 3, "error_cases": 0, "not_applicable_cases": 6, "status": "completed",
                       "task_results": [{"task_id": "math_canonical", "category": "math", "status": "passed",
                                         "response_text": "42"},
                                        {"task_id": "structured_json", "category": "structured_output", "status": "failed",
                                         "details": "invalid JSON", "response_text": "{oops"}]})
    for name, acc in evals:
        fn = f"eval_{name}_{model}_hf.json"
        models.append({"benchmark_type": "eval", "eval_name": name, "accuracy": acc, "total_questions": 4,
                       "correct_answers": 2, "leaderboard_comparable": False, "sample_traces_omitted": {"file": fn, "count": 2}})
        (d / fn).write_text(json.dumps({"eval_name": name, "accuracy": acc, "sample_traces": [
            {"sample_idx": 0, "messages": [{"role": "user", "content": "2+2?"}], "gold_answer": "4", "response_text": "4",
             "is_correct": True, "category": "arith"},
            {"sample_idx": 1, "messages": [{"role": "user", "content": "3*3?"}], "gold_answer": "9", "response_text": "6",
             "is_correct": False, "category": "arith", "finish_reason": "stop"}]}))
    if perf:
        models.append({"benchmark_type": "serving", "model_short": model, "median_tpot_ms": perf[0], "median_ttft_ms": perf[1],
                       "completed_requests": 12})
    (d / "summary.json").write_text(json.dumps({"models": models, "failures": []}))
    (d / "gbtui.json").write_text(json.dumps({"run_id": rid, "status": status, "started": 1000.0 + len(rid), "finished": 1100.0,
                                              "spec": {"tier": "evals" if evals else "golden", "endpoint": {"name": model}}}))
    return d


class TestArgs(unittest.TestCase):
    def test_golden(self):
        a = runner.gbench_args(runner.RunSpec(endpoint=ep(), tier="golden"), "/work/runs/x")
        self.assertIn("--golden-only", a)
        self.assertEqual(a[a.index("--remote-endpoint") + 1], f"http://{sandbox.host_alias()}:8080/v1")
        self.assertEqual(a[a.index("--tokenizer") + 1], "LiquidAI/LFM3-1.7B")

    def test_evals(self):
        a = runner.gbench_args(runner.RunSpec(endpoint=ep(), tier="evals", evals=["gsm8k", "ifeval"], eval_limit=5,
                                              thinking=True, temperature="0.2"), "/r")
        s = " ".join(a)
        for frag in ("--evals-only --evals gsm8k ifeval", "--eval-limit 5", "--max-output-tokens 4096", "--eval-thinking",
                     "--temperature 0.2"):
            self.assertIn(frag, s)

    def test_perf_and_preflight(self):
        perf_args = runner.gbench_args(runner.RunSpec(endpoint=ep(), tier="perf", thinking=True, temperature="0.5"), "/r")
        self.assertIn("--personal", perf_args)
        self.assertNotIn("--text-only", perf_args)
        self.assertNotIn("--eval-thinking", perf_args)
        self.assertNotIn("--temperature", perf_args)
        self.assertIn("--text-only", runner.gbench_args(runner.RunSpec(endpoint=ep(gguf="/m.gguf"), tier="perf"), "/r"))
        a = runner.gbench_args(runner.RunSpec(endpoint=ep(), tier="preflight", evals=["bfcl"]), "/r")
        self.assertIn("--preflight", a)
        self.assertNotIn("--eval-limit", a)

    def test_network_rules_and_endpoint_url(self):
        rules = sandbox.allow_for("http://127.0.0.1:8080/v1")
        self.assertIn(f"{sandbox.host_alias()}:8080", rules)
        self.assertIn("huggingface.co", rules)
        self.assertEqual(sandbox.allow_for("https://api.example.com/v1")[:2], ["api.example.com:443", "api.example.com:80"])
        self.assertTrue(config.Endpoint(name="v6", base_url="http://[::1]:8080/v1").is_local)
        self.assertEqual(config.Endpoint(name="tls", base_url="https://api.example.com/v1").port, 443)
        self.assertEqual(sandbox.sandbox_url("http://localhost"), f"http://{sandbox.host_alias()}")


class TestResults(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        fake_run(self.root, "20261124-100000-a-golden", "a", golden=70.0, perf=(20.0, 150.0))
        fake_run(self.root, "20261124-110000-a-evals", "a", evals=[("gsm8k", 50.0), ("ifeval", 25.0)])
        fake_run(self.root, "20261124-120000-b-evals", "b", evals=[("gsm8k", 75.0)])
        fake_run(self.root, "20261124-130000-b-failed", "b", evals=[("gsm8k", 5.0)], status="failed")

    def test_parse(self):
        runs = R.list_runs(self.root)
        self.assertEqual(len(runs), 4)
        g = next(r for r in runs if r.id.endswith("golden"))
        names = {s.name: s for s in g.scores}
        self.assertEqual(names["golden_set"].value, 70.0)
        self.assertAlmostEqual(names["decode tok/s"].value, 50.0)
        self.assertAlmostEqual(names["TTFT"].value, 0.15)
        items = R.items(g)
        self.assertEqual([i["correct"] for i in items], [True, False])

    def test_items_eval(self):
        run = R.load_run(self.root / "20261124-110000-a-evals")
        gsm = next(s for s in run.scores if s.name == "gsm8k")
        its = R.items(run, gsm)
        self.assertEqual(len(its), 2)
        self.assertIn("3*3?", its[1]["prompt"])
        self.assertFalse(its[1]["correct"])

    def test_leaderboard(self):
        lb = leaderboard.build(R.list_runs(self.root))
        self.assertEqual([m["model"] for m in lb["models"]], ["b", "a"])   # b: 75; a: (70+50+25)/3
        self.assertAlmostEqual(lb["models"][1]["gscore"], (70 + 50 + 25) / 3)
        self.assertEqual(lb["models"][0]["scores"]["gsm8k"].value, 75.0)    # the failed run is ignored
        md = leaderboard.to_markdown(lb)
        self.assertIn("| 1 | b |", md)
        self.assertIn("<table>", leaderboard.to_html(lb))

    def test_export(self):
        out = Path(tempfile.mkdtemp()) / "ds"
        export.export(R.list_runs(self.root), out, "jsonl", False, lambda _m: None)
        rows = [json.loads(x) for x in (out / "data" / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2 + 2 * 3 + 2)
        self.assertTrue(all("model" in r and "correct" in r for r in rows))
        self.assertIn("configs:", (out / "README.md").read_text())
        out2 = Path(tempfile.mkdtemp()) / "ds"
        export.export(R.list_runs(self.root), out2, "jsonl", True, lambda _m: None)
        fails = (out2 / "data" / "samples.jsonl").read_text().splitlines()
        self.assertTrue(all(not json.loads(x)["correct"] for x in fails))

    def test_leaderboard_preflight_and_escaping(self):
        fake_run(self.root, "20261124-140000-preflight-only", "pre|model", status="ok")
        lb = leaderboard.build(R.list_runs(self.root))
        self.assertNotIn("pre|model", [m["model"] for m in lb["models"]])
        # comparable_only filters out non-comparable evals
        lb_cmp = leaderboard.build(R.list_runs(self.root), comparable_only=True)
        self.assertEqual([m["model"] for m in lb_cmp["models"]], ["a"])
        # Markdown escaping for pipe in model name & HTML script safety + charts
        fake_run(self.root, "20261124-150000-pipe", "m|1</script>", golden=80.0)
        lb_all = leaderboard.build(R.list_runs(self.root))
        md = leaderboard.to_markdown(lb_all)
        self.assertIn(r"m\|1", md)
        html_out = leaderboard.to_html(lb_all, "aurora")
        self.assertIn("<table>", html_out)
        self.assertIn("renderRadar()", html_out)
        self.assertIn("renderBars()", html_out)
        self.assertNotIn("m|1</script>", html_out)

    def test_quantization_dimension_end_to_end(self):
        # 1. Quant detection & splitting across naming conventions
        self.assertEqual(config.split_model_quant("LFM3-1.7B_Q4_K_M.gguf"), ("LFM3-1.7B", "Q4_K_M"))
        self.assertEqual(config.split_model_quant("LFM3-1.7B-Q8_0.gguf"), ("LFM3-1.7B", "Q8_0"))
        self.assertEqual(config.split_model_quant("gemma-3-4b-UD-Q4_K_XL.gguf"), ("gemma-3-4b", "UD-Q4_K_XL"))
        self.assertEqual(config.split_model_quant("qwen3:4b-q4_K_M"), ("qwen3:4b", "Q4_K_M"))
        self.assertEqual(config.split_model_quant("Qwen2.5-1.5B"), ("Qwen2.5-1.5B", ""))
        self.assertEqual(config.split_model_quant("gemma-3-4b"), ("gemma-3-4b", ""))
        self.assertLess(config.quant_bits("Q4_K_M"), config.quant_bits("Q8_0"))
        self.assertLess(config.quant_bits("Q8_0"), config.quant_bits("BF16"))

        # 2. Endpoint auto-detects quant & run_id embeds quant when omitted from name
        ep_q4 = config.Endpoint(name="lfm3-1.7b", gguf="/models/LFM3-1.7B_Q4_K_M.gguf")
        self.assertEqual(ep_q4.effective_quant, "Q4_K_M")
        self.assertEqual(ep_q4.base_model, "lfm3-1.7b")
        self.assertIn("Q4_K_M", runner.RunSpec(endpoint=ep_q4, tier="golden").run_id())

        # 3. Same base model evaluated at Q4_K_M and Q8_0 stays distinct (even with identical endpoint name!)
        qroot = Path(tempfile.mkdtemp())

        def make_q_run(rid, ep_dict, golden, gsm8k, tpot_ms):
            d = qroot / rid
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"models": [
                {"benchmark_type": "golden", "accuracy_percent": golden, "applicable_tasks": 10, "passed_cases": 8,
                 "failed_cases": 2, "error_cases": 0, "not_applicable_cases": 0, "status": "completed"},
                {"benchmark_type": "eval", "eval_name": "gsm8k", "accuracy": gsm8k, "total_questions": 20,
                 "correct_answers": 15, "leaderboard_comparable": True},
                {"benchmark_type": "serving", "median_tpot_ms": tpot_ms, "median_ttft_ms": 120.0, "completed_requests": 8},
            ], "failures": []}))
            (d / "gbtui.json").write_text(json.dumps({
                "run_id": rid, "status": "ok", "started": 2000.0 + len(rid), "finished": 2100.0,
                "spec": {"tier": "evals", "endpoint": ep_dict},
            }))

        make_q_run("20261127-100000-q4", {"name": "LFM3-1.7B_Q4_K_M", "gguf": "/m/LFM3-1.7B_Q4_K_M.gguf"}, 78.0, 72.0, 6.0)
        make_q_run("20261127-110000-q8", {"name": "LFM3-1.7B_Q8_0", "gguf": "/m/LFM3-1.7B_Q8_0.gguf"}, 86.0, 82.0, 10.0)
        # Reused endpoint name with different GGUF quant must NOT overwrite Q8_0:
        make_q_run("20261127-120000-bf16", {"name": "LFM3-1.7B", "gguf": "/m/LFM3-1.7B-BF16.gguf"}, 88.0, 84.0, 18.0)

        runs = R.list_runs(qroot)
        lb = leaderboard.build(runs)
        self.assertEqual(len(lb["models"]), 3)
        self.assertEqual(lb["quants"], ["Q4_K_M", "Q8_0", "BF16"])
        self.assertEqual(lb["families"], ["LFM3-1.7B"])
        self.assertEqual([(m["base_model"], m["quant"]) for m in lb["models"]],
                         [("LFM3-1.7B", "BF16"), ("LFM3-1.7B", "Q8_0"), ("LFM3-1.7B", "Q4_K_M")])

        md = leaderboard.to_markdown(lb)
        self.assertIn("| # | Model | Quant | gScore |", md)
        self.assertIn("| LFM3-1.7B | Q4_K_M |", md)
        self.assertIn("| LFM3-1.7B | Q8_0 |", md)

        html_q = leaderboard.to_html(lb, "aurora")
        self.assertIn("By Quant", html_q)
        self.assertIn("quant_efficiency", html_q)
        self.assertIn("Q4_K_M", html_q)
        self.assertIn("Q8_0", html_q)

        # 4. Export includes base_model & quant metadata columns
        exp_out = Path(tempfile.mkdtemp()) / "ds_q"
        export.export(runs, exp_out, "jsonl", False, lambda _m: None)
        score_rows = [json.loads(x) for x in (exp_out / "data" / "scores.jsonl").read_text().splitlines()]
        self.assertEqual({r["quant"] for r in score_rows}, {"Q4_K_M", "Q8_0", "BF16"})
        self.assertEqual({r["base_model"] for r in score_rows}, {"LFM3-1.7B"})


if __name__ == "__main__":
    unittest.main()
