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
        self.assertIn("--personal", runner.gbench_args(runner.RunSpec(endpoint=ep(), tier="perf"), "/r"))
        a = runner.gbench_args(runner.RunSpec(endpoint=ep(), tier="preflight", evals=["bfcl"]), "/r")
        self.assertIn("--preflight", a)
        self.assertNotIn("--eval-limit", a)

    def test_network_rules(self):
        rules = sandbox.allow_for("http://127.0.0.1:8080/v1")
        self.assertIn(f"{sandbox.host_alias()}:8080", rules)
        self.assertIn("huggingface.co", rules)
        self.assertEqual(sandbox.allow_for("https://api.example.com/v1")[:2], ["api.example.com:443", "api.example.com:80"])


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


if __name__ == "__main__":
    unittest.main()
