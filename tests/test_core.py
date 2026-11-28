"""Unit tests that need neither Docker nor a model: gbench's command line, container hardening, result parsing, leaderboard and export."""
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

from gbtui import config, export, harbor_runner, harness_hub, leaderboard, results as R, runner, sandbox, servers  # noqa: E402


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

    def test_docker_hardening_and_lifecycle_helpers(self):
        # Online run container: read-only, cap-drop=ALL, no-new-privileges, RAM/CPU/PID caps, host-gateway, /work bind
        argv = sandbox.run_argv("gbtui-run-1-abc", 4, 6, sandbox.allow_for("http://127.0.0.1:8080/v1"))
        self.assertEqual(argv[:3], ["docker", "run", "-d"])
        for flag in ("--read-only", "--cap-drop=ALL", "no-new-privileges:true", "4g", "6", str(sandbox.PIDS_LIMIT),
                     f"{sandbox.host_alias()}:host-gateway", "GBENCH_SANDBOX=none",
                     f"{sandbox.work_dir()}:{sandbox.CONTAINER_WORK}:rw"):
            self.assertIn(flag, argv)
        self.assertNotIn("--network", argv)
        self.assertEqual(argv[-3:], [sandbox.IMAGE, "sleep", "infinity"])

        # Offline container (catalog / parquet): air-gapped with --network none
        offline = sandbox.run_argv("gbtui-run-1-off", 2, 2, [])
        self.assertIn("--network", offline)
        self.assertEqual(offline[offline.index("--network") + 1], "none")
        self.assertNotIn("--add-host", offline)

        # exec_argv passes env & command directly without shell interpolation
        ex = sandbox.exec_argv("gbtui-run-1-abc", ["gbench", "--golden-only"], env={"HF_HOME": "/work/hf"})
        self.assertEqual(ex, ["docker", "exec", "-e", "HF_HOME=/work/hf", "gbtui-run-1-abc", "gbench", "--golden-only"])

        # Secret environment variables are stripped before invoking docker
        old_tok = os.environ.get("FAKE_TEST_API_TOKEN")
        os.environ["FAKE_TEST_API_TOKEN"] = "super-secret"
        try:
            self.assertNotIn("FAKE_TEST_API_TOKEN", sandbox._env())
        finally:
            if old_tok is None:
                os.environ.pop("FAKE_TEST_API_TOKEN", None)
            else:
                os.environ["FAKE_TEST_API_TOKEN"] = old_tok

        # Dockerfile contains base image, /work, nltk_data and world-readable chmod for non-root --read-only
        df = sandbox.dockerfile()
        self.assertIn(f"FROM {sandbox.BASE_IMAGE}", df)
        self.assertIn(f"WORKDIR {sandbox.CONTAINER_WORK}", df)
        self.assertIn("punkt_tab", df)
        self.assertIn("chmod -R a+rX", df)

        # Preview uses /work container path
        self.assertIn(f"--results-dir {sandbox.CONTAINER_WORK}/runs/<id>",
                      runner.command_preview(runner.RunSpec(endpoint=ep(), tier="golden")))


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

    def test_lora_and_harness_adapter_detection_and_servers(self):
        # Harness detection without false positives on 'api' / 'pipeline'
        self.assertEqual(config.detect_harness("LFM3-1.7B-harbor-pi-lora-F16.gguf"), "pi")
        self.assertEqual(config.detect_harness("LFM3-1.7B-harbor-opencode-lora-F16.gguf"), "opencode")
        self.assertEqual(config.detect_harness("LFM3-1.7B-harbor-hermes-lora-F16.gguf"), "hermes")
        self.assertEqual(config.detect_harness("Qwen3.8-27B-super-adapter-GGUF"), "super-adapter")
        self.assertEqual(config.detect_harness("openai-api-pipeline-v2"), "")

        tmp = Path(tempfile.mkdtemp())
        base_f = tmp / "LFM3-1.7B-Q8_0.gguf"
        lora_f = tmp / "LFM3-1.7B-harbor-pi-lora-F16.gguf"
        base_f.write_bytes(b"0" * 10000)
        lora_f.write_bytes(b"1" * 4000)

        ep_base = config.Endpoint(name="lfm3-1.7b", gguf=str(base_f), ctx=16384)
        ep_pi = config.Endpoint(name="lfm3-1.7b-pi-lora", gguf=str(base_f), lora=str(lora_f), ctx=16384)
        ep_scaled = config.Endpoint(name="lfm3-1.7b-pi-075", gguf=str(base_f), lora=str(lora_f), lora_scale=0.75, ctx=65536)

        self.assertEqual(ep_base.effective_adapter, "")
        self.assertEqual(ep_base.ctx_warning, "")
        self.assertEqual(ep_pi.effective_harness, "pi")
        self.assertEqual(ep_pi.effective_adapter, "pi-lora")
        self.assertEqual(ep_pi.base_model, "lfm3-1.7b")
        self.assertIn("32768", ep_pi.ctx_warning)
        self.assertEqual(ep_scaled.ctx_warning, "")

        # llama-server command uses --lora vs --lora-scaled and estimate_bytes includes LoRA size
        cmd_base = servers.llama_cmd(ep_base)
        self.assertNotIn("--lora", cmd_base)
        cmd_pi = servers.llama_cmd(ep_pi)
        self.assertIn("--lora", cmd_pi)
        self.assertEqual(cmd_pi[cmd_pi.index("--lora") + 1], str(lora_f))
        cmd_sc = servers.llama_cmd(ep_scaled)
        self.assertIn("--lora-scaled", cmd_sc)
        self.assertEqual(cmd_sc[cmd_sc.index("--lora-scaled") + 1: cmd_sc.index("--lora-scaled") + 3], [str(lora_f), "0.75"])
        self.assertGreater(servers.estimate_bytes(ep_pi), servers.estimate_bytes(ep_base))

    def test_harness_hub_parsing_and_compatibility(self):
        raw = [
            {
                "id": "harness-hub/LFM3-1.7B-pi-GGUF", "downloads": 411, "likes": 11,
                "tags": ["gguf", "llama.cpp", "lora", "pi", "base_model:harness-hub/LFM3-1.7B-pi"],
                "siblings": [{"rfilename": "LFM3-1.7B-harbor-pi-lora-F16.gguf"}, {"rfilename": "README.md"}],
            },
            {
                "id": "harness-hub/LFM3-1.7B-opencode-GGUF", "downloads": 303, "likes": 10,
                "tags": ["gguf", "llama.cpp", "lora", "opencode"],
                "siblings": [{"rfilename": "LFM3-1.7B-harbor-opencode-lora-F16.gguf"}],
            },
            {
                "id": "harness-hub/LFM3-1.7B-pi", "downloads": 152, "likes": 11,
                "tags": ["peft", "safetensors", "lora", "pi"],
                "siblings": [{"rfilename": "adapter_model.safetensors"}],
            },
            {
                "id": "harness-hub/Qwen3.8-27B-super-adapter-GGUF", "downloads": 900, "likes": 50,
                "tags": ["gguf", "llama.cpp", "lora"],
                "siblings": [{"rfilename": "Qwen3.8-27B-super-adapter-lora-F16.gguf"}],
            },
        ]
        parsed = harness_hub.parse_gguf_models(raw)
        self.assertEqual(len(parsed), 3)  # PEFT-only repo skipped
        lfm = harness_hub.compatible_adapters("LFM3-1.7B", parsed)
        self.assertEqual([x["harness"] for x in lfm], ["pi", "opencode"])
        self.assertEqual([x["adapter_name"] for x in lfm], ["pi-lora", "opencode-lora"])
        qwen = harness_hub.compatible_adapters("Qwen3.8-27B", parsed)
        self.assertEqual(len(qwen), 1)
        self.assertEqual(qwen[0]["adapter_name"], "super-adapter")

    def test_harbor_runner_and_adapter_harness_matrix(self):
        # 1. Command building
        ep_pi = config.Endpoint(name="lfm3-pi", base_url="http://127.0.0.1:8080/v1", model="lfm3-pi",
                                gguf="/m/LFM3-1.7B-Q8_0.gguf", lora="/l/LFM3-1.7B-harbor-pi-lora-F16.gguf", ctx=65536)
        args = harbor_runner.build_args(ep_pi, "pi", "terminal-bench@2.0", Path("/tmp/jobs"), "job1",
                                        limit=2, tasks=["hello*"], concurrency=1, max_output_tokens=4096)
        s_args = " ".join(args)
        for frag in ("--dataset terminal-bench@2.0", "--agent pi", "--model openai/lfm3-pi",
                     f"--base-url http://{sandbox.host_alias()}:8080/v1", "--api-format openai_chat_completions",
                     "--env docker", "--n-tasks 2", "--include-task-name hello*", "--ak context_window=65536"):
            self.assertIn(frag, s_args)
        # OpenCode uses custom `local/<model>` provider prefix so @ai-sdk/openai-compatible calls chat completions
        oc_args = harbor_runner.build_args(ep_pi, "opencode", "terminal-bench@2.0", Path("/tmp/jobs"), "job2", limit=1)
        self.assertIn("local/lfm3-pi", oc_args)

        # 2. Harbor job directory parsing (result.json + ATIF trajectory.json)
        jdir = Path(tempfile.mkdtemp())
        t1 = jdir / "task-alpha__abc123"
        (t1 / "agent").mkdir(parents=True)
        (t1 / "result.json").write_text(json.dumps({
            "task_name": "task-alpha", "trial_name": "task-alpha__abc123",
            "verifier_result": {"rewards": {"reward": 1.0}},
            "agent_result": {"n_input_tokens": 1200, "n_output_tokens": 300},
            "agent_execution": {"started_at": "2026-11-28T10:00:00Z", "finished_at": "2026-11-28T10:00:12Z"},
        }))
        (t1 / "agent" / "trajectory.json").write_text(json.dumps({
            "steps": [
                {"source": "user", "message": "Fix the script"},
                {"source": "agent", "message": "Running ls", "tool_calls": [{"function_name": "bash", "arguments": {"cmd": "ls"}}]},
                {"source": "agent", "message": "Done", "tool_calls": [{"function_name": "edit", "arguments": {}}]},
            ],
        }))
        t2 = jdir / "task-beta__def456"
        (t2 / "agent").mkdir(parents=True)
        (t2 / "result.json").write_text(json.dumps({
            "task_name": "task-beta", "trial_name": "task-beta__def456",
            "verifier_result": {"rewards": {"reward": 0.0}},
            "agent_result": {"n_input_tokens": 2000, "n_output_tokens": 500},
        }))
        (t2 / "agent" / "trajectory.json").write_text(json.dumps({
            "steps": [
                {"source": "agent", "message": "Trying", "tool_calls": [{"function_name": "bash"}, {"function_name": "bash"}, {"function_name": "read"}, {"function_name": "bash"}]},
            ],
        }))
        summary, trials = harbor_runner.parse_job(jdir, "pi", "terminal-bench@2.0")
        ent = summary["models"][0]
        self.assertEqual(ent["benchmark_type"], "harness")
        self.assertEqual(ent["resolved_tasks"], 1)
        self.assertEqual(ent["total_tasks"], 2)
        self.assertAlmostEqual(ent["resolved_rate"], 50.0)
        self.assertAlmostEqual(ent["avg_tokens"], 2000.0)
        self.assertAlmostEqual(ent["avg_tool_calls"], 3.0)
        self.assertEqual(len(trials), 2)

        # 3. Full Adapter × Harness leaderboard + matrix + HTML + export
        mroot = Path(tempfile.mkdtemp())

        def make_h_run(rid, ep_dict, harness, resolved_pct, res, tot, avg_tok, avg_tc):
            d = mroot / rid
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"models": [{
                "benchmark_type": "harness", "harness": harness, "dataset": "terminal-bench@2.0",
                "dataset_short": "terminal-bench", "resolved_rate": resolved_pct, "resolved_tasks": res,
                "total_tasks": tot, "error_tasks": 0, "avg_tokens": avg_tok, "avg_tool_calls": avg_tc,
                "status": "completed", "trials_file": "harbor_trials.json",
            }], "failures": []}))
            (d / "harbor_trials.json").write_text(json.dumps(trials))
            (d / "gbtui.json").write_text(json.dumps({
                "run_id": rid, "backend": "harbor", "status": "ok", "started": 3000.0 + len(rid), "finished": 3120.0,
                "spec": {"tier": "harness", "harness": harness, "harness_dataset": "terminal-bench@2.0", "endpoint": ep_dict},
            }))

        ep_b = {"name": "lfm3-1.7b", "gguf": "/m/LFM3-1.7B-Q8_0.gguf", "quant": "Q8_0"}
        ep_p = {"name": "lfm3-1.7b-pi-lora", "gguf": "/m/LFM3-1.7B-Q8_0.gguf", "quant": "Q8_0",
                "lora": "/l/LFM3-1.7B-harbor-pi-lora-F16.gguf", "adapter_name": "pi-lora", "target_harness": "pi"}
        ep_o = {"name": "lfm3-1.7b-opencode-lora", "gguf": "/m/LFM3-1.7B-Q8_0.gguf", "quant": "Q8_0",
                "lora": "/l/LFM3-1.7B-harbor-opencode-lora-F16.gguf", "adapter_name": "opencode-lora", "target_harness": "opencode"}

        make_h_run("20261128-100001-b-pi", ep_b, "pi", 25.0, 1, 4, 5200.0, 9.0)
        make_h_run("20261128-100002-b-oc", ep_b, "opencode", 25.0, 1, 4, 6100.0, 11.0)
        make_h_run("20261128-100003-p-pi", ep_p, "pi", 75.0, 3, 4, 3100.0, 5.5)
        make_h_run("20261128-100004-p-oc", ep_p, "opencode", 50.0, 2, 4, 4800.0, 8.0)
        make_h_run("20261128-100005-o-pi", ep_o, "pi", 50.0, 2, 4, 4500.0, 7.5)
        make_h_run("20261128-100006-o-oc", ep_o, "opencode", 75.0, 3, 4, 3400.0, 6.0)

        runs = R.list_runs(mroot)
        self.assertEqual(len(runs), 6)
        lb = leaderboard.build(runs)
        self.assertEqual(len(lb["models"]), 6)  # all 6 (base, quant, adapter, harness) cells distinct!
        self.assertEqual(lb["adapters"], ["base", "opencode-lora", "pi-lora"])
        self.assertEqual(lb["harnesses"], ["opencode", "pi"])
        mat = lb["matrix"]
        self.assertEqual(mat["adapters"], ["base", "opencode-lora", "pi-lora"])
        self.assertEqual(mat["harnesses"], ["opencode", "pi"])
        self.assertEqual(mat["cells"]["pi-lora||pi"]["resolved_pct"], 75.0)
        self.assertEqual(mat["cells"]["opencode-lora||opencode"]["avg_tool_calls"], 6.0)

        md = leaderboard.to_markdown(lb)
        self.assertIn("## Adapter × Harness Matrix", md)
        self.assertIn("| pi-lora |", md)
        html_out = leaderboard.to_html(lb, "aurora")
        self.assertIn("Adapter × Harness Heatmap Matrix", html_out)
        self.assertIn("renderMatrix()", html_out)
        self.assertIn("pi-lora", html_out)
        self.assertIn("opencode-lora", html_out)

        # Export includes adapter, target_harness, harness and per-trial Harbor items
        exp_out = Path(tempfile.mkdtemp()) / "ds_h"
        export.export(runs, exp_out, "jsonl", False, lambda _m: None)
        s_rows = [json.loads(x) for x in (exp_out / "data" / "scores.jsonl").read_text().splitlines()]
        t_rows = [json.loads(x) for x in (exp_out / "data" / "samples.jsonl").read_text().splitlines()]
        self.assertEqual({r["adapter"] for r in s_rows}, {"base", "pi-lora", "opencode-lora"})
        self.assertEqual({r["harness"] for r in s_rows}, {"pi", "opencode"})
        self.assertEqual(len(t_rows), 12)  # 6 runs × 2 trials each

    def test_gguf_binary_header_reader(self):
        import struct
        from gbtui import gguf_meta

        def pack_str(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        def pack_kv_str(k: str, v: str) -> bytes:
            return pack_str(k) + struct.pack("<I", 8) + pack_str(v)

        def pack_kv_u32(k: str, v: int) -> bytes:
            return pack_str(k) + struct.pack("<I", 4) + struct.pack("<I", v)

        def write_fake_gguf(path: Path, arch: str, name: str, ftype: int, ctx: int, is_lora: bool = False):
            kvs = [
                pack_kv_str("general.architecture", arch),
                pack_kv_str("general.name", name),
                pack_kv_u32("general.file_type", ftype),
                pack_kv_u32(f"{arch}.context_length", ctx),
            ]
            if is_lora:
                kvs.append(pack_kv_str("general.type", "adapter"))
            # 1 tensor of shape [4096, 4096] -> 16,777,216 params
            tensor = pack_str("blk.0.attn_q.weight") + struct.pack("<IQQIQ", 2, 4096, 4096, 0, 0)
            hdr = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs) + tensor + b"\x00" * 64
            path.write_bytes(hdr)

        d = Path(tempfile.mkdtemp())
        m_path = d / "LFM3-1.7B-Q4_K_M.gguf"
        l_path = d / "LFM3-1.7B-hermes-lora-F16.gguf"
        write_fake_gguf(m_path, "lfm2", "LFM3-1.7B", 15, 65536, is_lora=False)
        write_fake_gguf(l_path, "lfm2", "LFM3-1.7B-hermes", 1, 65536, is_lora=True)

        info = gguf_meta.read_gguf(m_path)
        self.assertIsNotNone(info)
        self.assertEqual(info.architecture, "lfm2")
        self.assertEqual(info.name, "LFM3-1.7B")
        self.assertEqual(info.quant, "Q4_K_M")
        self.assertEqual(info.context_length, 65536)
        self.assertEqual(info.param_count, 4096 * 4096)
        self.assertFalse(info.is_lora)

        l_info = gguf_meta.read_gguf(l_path)
        self.assertIsNotNone(l_info)
        self.assertTrue(l_info.is_lora)
        self.assertEqual(l_info.quant, "F16")

        bases, loras = gguf_meta.scan_folder(d)
        self.assertEqual(len(bases), 1)
        self.assertEqual(len(loras), 1)

    def test_compare_diagnostics_notes_and_report(self):
        from gbtui import compare as CMP, report

        root = Path(tempfile.mkdtemp())

        def make_eval_run(rid: str, quant: str, samples: list[dict], speed: float):
            d = root / rid
            d.mkdir(parents=True)
            passed = sum(1 for s in samples if s["is_correct"])
            pct = passed * 100.0 / len(samples)
            (d / "summary.json").write_text(json.dumps({
                "models": [
                    {"benchmark_type": "eval", "eval_name": "gsm8k", "accuracy": pct, "total_questions": len(samples),
                     "correct_answers": passed, "status": "completed",
                     "sample_traces_omitted": {"file": "gsm8k_samples.json"}},
                    {"benchmark_type": "personal_perf", "input_length": 512, "tpot_p50_ms": 1000.0 / speed, "ttft_p50_ms": 250.0},
                ],
                "failures": [],
            }))
            (d / "gsm8k_samples.json").write_text(json.dumps({"sample_traces": samples}))
            (d / "gbtui.json").write_text(json.dumps({
                "run_id": rid, "status": "ok", "started": 5000.0 + len(rid), "finished": 5060.0,
                "spec": {"tier": "evals", "endpoint": {"name": f"m-{quant}", "model": f"m-{quant}", "quant": quant}},
            }))
            return R.load_run(d)

        ra = make_eval_run("20261128-100001-q8", "Q8_0", [
            {"sample_key": "s1", "messages": [{"role": "user", "content": "2+2?"}], "response_text": "4", "is_correct": True, "completion_tokens": 20, "finish_reason": "stop"},
            {"sample_key": "s2", "messages": [{"role": "user", "content": "3+3?"}], "response_text": "6", "is_correct": True, "completion_tokens": 30, "finish_reason": "stop"},
            {"sample_key": "s3", "messages": [{"role": "user", "content": "4+4?"}], "response_text": "9", "is_correct": False, "completion_tokens": 40, "finish_reason": "stop", "error": "wrong"},
        ], speed=95.0)
        rb = make_eval_run("20261128-100002-q4", "Q4_K_M", [
            {"sample_key": "s1", "messages": [{"role": "user", "content": "2+2?"}], "response_text": "4", "is_correct": True, "completion_tokens": 22, "finish_reason": "stop"},
            {"sample_key": "s2", "messages": [{"role": "user", "content": "3+3?"}], "response_text": "thinking...", "is_correct": False, "completion_tokens": 4096, "finish_reason": "length", "error": "truncated"},
            {"sample_key": "s3", "messages": [{"role": "user", "content": "4+4?"}], "response_text": "8", "is_correct": True, "completion_tokens": 28, "finish_reason": "stop"},
        ], speed=140.0)

        # Notes, tags & archive persistence
        R.save_meta(rb, notes="ctx 32k + flash-attn", tags=["q4", "fast"], archived=True)
        rb_reloaded = R.load_run(rb.path)
        self.assertEqual(rb_reloaded.notes, "ctx 32k + flash-attn")
        self.assertEqual(rb_reloaded.tags, ["q4", "fast"])
        self.assertTrue(rb_reloaded.archived)
        self.assertEqual(len(R.list_runs(root, include_archived=False)), 1)
        self.assertEqual(len(R.list_runs(root, include_archived=True)), 2)
        R.save_meta(rb, archived=False)

        # Diagnostics: failure taxonomy + Token Economy
        diag_b = CMP.diagnose_run(rb)
        self.assertEqual(diag_b["causes"]["truncated"], 1)
        self.assertEqual(diag_b["passed"], 2)
        self.assertEqual(diag_b["failed"], 1)
        self.assertAlmostEqual(diag_b["avg_tokens_pass"], 25.0)
        self.assertAlmostEqual(diag_b["avg_tokens_fail"], 4096.0)
        self.assertIsNotNone(diag_b["gscore_per_1k_tokens"])

        # Compare A vs B: 1 regression (s2), 1 improvement (s3), 1 both_pass (s1)
        diff = CMP.compare_runs(ra, rb)
        self.assertEqual(diff.regressions, 1)
        self.assertEqual(diff.improvements, 1)
        self.assertEqual(diff.both_pass, 1)

        # Pareto HTML & Report Markdown/HTML
        runs = R.list_runs(root)
        lb = leaderboard.build(runs)
        html_lb = leaderboard.to_html(lb, "tokyo")
        self.assertIn("renderPareto()", html_lb)
        self.assertIn("Pareto Frontier", html_lb)

        md_rep = report.build_markdown(runs)
        self.assertIn("Quantization Degradation", md_rep)
        self.assertIn("Failure Taxonomy & Token Economy", md_rep)
        self.assertIn("Regressions (Pass → Fail)", md_rep)
        html_rep = report.build_html(runs, theme_name="dracula")
        self.assertIn("gbench-tui Evaluation Report", html_rep)

    def test_code_mode_metrics_presets_sweep_and_harness_lens(self):
        from gbtui import harness_lens

        # 1. Code Mode vs Normal classification & Hermes CLI flags
        self.assertEqual(harbor_runner.classify_tool_call("execute_code"), "code_mode")
        self.assertEqual(harbor_runner.classify_tool_call("bash"), "code_mode")
        self.assertEqual(harbor_runner.classify_tool_call("read_file"), "normal")
        self.assertEqual(harbor_runner.classify_tool_call("edit"), "normal")

        ep = config.Endpoint(name="lfm3-hermes", base_url="http://127.0.0.1:8080/v1", model="lfm3-hermes", ctx=65536)
        cm_args = harbor_runner.build_args(ep, "hermes", "terminal-bench@2.0", Path("/tmp/j"), "job_cm",
                                           limit=1, tool_mode="code_mode")
        self.assertIn("toolsets=codemode,terminal,file", " ".join(cm_args))
        self.assertIn("--extra-instruction", cm_args)

        # Parse ATIF with mixed execute_code + normal tools + cached tokens + tool error
        jdir = Path(tempfile.mkdtemp())
        t1 = jdir / "task1__111"
        (t1 / "agent").mkdir(parents=True)
        (t1 / "result.json").write_text(json.dumps({
            "task_name": "task1", "trial_name": "task1__111",
            "verifier_result": {"rewards": {"reward": 1.0}},
            "agent_result": {"n_input_tokens": 10000, "n_output_tokens": 1000, "n_cache_tokens": 7500},
        }))
        (t1 / "agent" / "trajectory.json").write_text(json.dumps({
            "steps": [
                {"source": "agent", "tool_calls": [{"function_name": "execute_code"}, {"function_name": "read_file"}]},
                {"source": "observation", "observation": {"results": [{"content": "Error: file not found"}]}},
                {"source": "agent", "tool_calls": [{"function_name": "execute_code"}]},
            ],
        }))
        summary, _trials = harbor_runner.parse_job(jdir, "hermes", "terminal-bench@2.0", tool_mode="code_mode")
        ent = summary["models"][0]
        self.assertEqual(ent["tool_mode"], "code_mode")
        self.assertAlmostEqual(ent["avg_code_mode_calls"], 2.0)
        self.assertAlmostEqual(ent["avg_normal_tool_calls"], 1.0)
        self.assertAlmostEqual(ent["code_mode_ratio_pct"], 66.7, places=1)
        self.assertAlmostEqual(ent["cache_hit_rate_pct"], 75.0)
        self.assertEqual(ent["tool_distribution"]["execute_code"], 2)

        # 2. Presets, Quant Sweep builder & Duration estimation
        s = config.Settings()
        eps = [config.Endpoint(name="m-q8", gguf="/tmp/a.gguf"), config.Endpoint(name="m-q4", gguf="/tmp/b.gguf")]
        sweep_specs = runner.build_sweep_specs(eps, "smoke", s, stop_between=True)
        self.assertEqual(len(sweep_specs), 4)  # 2 endpoints × 2 steps (golden + perf)
        self.assertFalse(sweep_specs[0].stop_server_after)
        self.assertTrue(sweep_specs[1].stop_server_after)
        self.assertTrue(sweep_specs[3].stop_server_after)
        # Duration estimate with a prior perf run measuring 100 tok/s
        fake_run = R.Run(
            id="r1", path=Path(tempfile.mkdtemp()),
            meta={"status": "ok", "started": 1.0, "finished": 2.0, "spec": {"tier": "perf", "endpoint": {"name": "m-q8", "model": "m-q8"}}},
            scores=[R.Score("perf", "decode tok/s (in512)", 100.0, "tok/s")],
        )
        est_s, est_src = runner.estimate_duration_seconds(sweep_specs[0], [fake_run])
        self.assertIsNotNone(est_s)
        self.assertGreater(est_s, 0)
        self.assertIn("tok/s", est_src)

        # 3. Harness Lens specs & launch command
        ids = {h.id for h in harness_lens.HARNESS_SPECS}
        self.assertTrue({"hermes", "pi", "opencode"} <= ids)
        plain = harness_lens.format_plain("hermes")
        self.assertIn("Hermes", plain)
        self.assertIn("codemode", plain)

    def test_distillation_export_chatml_sharegpt_and_harbor_trajectories(self):
        root = Path(tempfile.mkdtemp())
        # Eval run with 1 passing (with reasoning) + 1 failing sample
        ev_dir = root / "20261128-110001-teacher-evals"
        ev_dir.mkdir(parents=True)
        (ev_dir / "summary.json").write_text(json.dumps({
            "models": [{"benchmark_type": "eval", "eval_name": "gsm8k", "accuracy": 50.0,
                        "total_questions": 2, "correct_answers": 1, "status": "completed",
                        "sample_traces_omitted": {"file": "gsm8k.json"}}],
            "failures": [],
        }))
        (ev_dir / "gsm8k.json").write_text(json.dumps({"sample_traces": [
            {"sample_key": "q1", "messages": [{"role": "user", "content": "What is 6*7?"}],
             "reasoning": "6 times 7 equals 42.", "response_text": "42", "is_correct": True, "completion_tokens": 25},
            {"sample_key": "q2", "messages": [{"role": "user", "content": "What is 9*9?"}],
             "reasoning": "wrong math", "response_text": "80", "is_correct": False, "completion_tokens": 20},
        ]}))
        (ev_dir / "gbtui.json").write_text(json.dumps({
            "run_id": ev_dir.name, "status": "ok", "started": 7000.0, "finished": 7010.0,
            "spec": {"tier": "evals", "endpoint": {"name": "teacher-70b", "model": "teacher-70b", "quant": "Q8_0"}},
        }))

        # Harbor run with full multi-turn ATIF trajectory (system + user + tool_call + observation + final answer)
        hb_dir = root / "20261128-110002-teacher-harness-pi"
        trial_dir = hb_dir / "task-fix__abc" / "agent"
        trial_dir.mkdir(parents=True)
        (trial_dir / "trajectory.json").write_text(json.dumps({
            "steps": [
                {"source": "system", "message": "You are pi coding agent."},
                {"source": "user", "message": "Fix bug in app.py"},
                {"source": "agent", "reasoning_content": "Let's read app.py first", "message": "",
                 "tool_calls": [{"tool_call_id": "c1", "function_name": "read", "arguments": {"path": "app.py"}}],
                 "observation": {"results": [{"source_call_id": "c1", "content": "print(1/0)"}]}},
                {"source": "agent", "message": "Fixed the division by zero in app.py."},
            ],
        }))
        (hb_dir / "harbor_trials.json").write_text(json.dumps([
            {"task_name": "task-fix", "trial_name": "task-fix__abc", "reward": 1.0, "correct": True,
             "status": "passed", "prompt": "Fix bug in app.py", "response": "Fixed.", "tool_calls": 1},
        ]))
        (hb_dir / "summary.json").write_text(json.dumps({
            "models": [{"benchmark_type": "harness", "harness": "pi", "dataset": "terminal-bench@2.0",
                        "dataset_short": "terminal-bench", "resolved_rate": 100.0, "resolved_tasks": 1,
                        "total_tasks": 1, "status": "completed", "trials_file": "harbor_trials.json"}],
            "failures": [],
        }))
        (hb_dir / "gbtui.json").write_text(json.dumps({
            "run_id": hb_dir.name, "backend": "harbor", "status": "ok", "started": 7020.0, "finished": 7050.0,
            "spec": {"tier": "harness", "harness": "pi", "endpoint": {"name": "teacher-70b", "quant": "Q8_0"}},
        }))

        runs = R.list_runs(root)
        # 1. ChatML (Unsloth Studio / HF TRL) with rejection sampling (passed only) + think_tags
        out_chat = Path(tempfile.mkdtemp()) / "distill-chatml"
        export.export(runs, out_chat, "jsonl", False, lambda _m: None,
                      schema="chatml", split_filter="passed", reasoning_mode="think_tags")
        rows = [json.loads(ln) for ln in (out_chat / "data" / "train.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)  # q2 (failed) filtered out; q1 + Harbor trajectory kept!
        # Check eval trace has <think> tags
        ev_rec = next(r for r in rows if r["metadata"]["suite"] == "gsm8k")
        self.assertIn("<think>\n6 times 7 equals 42.\n</think>", ev_rec["messages"][-1]["content"])
        # Check Harbor multi-turn trajectory reconstructed with system, user, tool_calls, tool observation, final assistant
        hb_rec = next(r for r in rows if r["metadata"]["harness"] == "pi")
        roles = [m["role"] for m in hb_rec["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(hb_rec["messages"][2]["tool_calls"][0]["function"]["name"], "read")
        readme = (out_chat / "README.md").read_text()
        self.assertIn("Unsloth Studio", readme)
        self.assertIn("SFTTrainer", readme)

        # 2. ShareGPT with reasoning_content stripped
        out_sg = Path(tempfile.mkdtemp()) / "distill-sharegpt"
        export.export(runs, out_sg, "jsonl", False, lambda _m: None,
                      schema="sharegpt", split_filter="passed", reasoning_mode="strip")
        sg_rows = [json.loads(ln) for ln in (out_sg / "data" / "train.jsonl").read_text().splitlines()]
        self.assertEqual(len(sg_rows), 2)
        sg_ev = next(r for r in sg_rows if r["metadata"]["suite"] == "gsm8k")
        self.assertEqual(sg_ev["conversations"][0]["from"], "human")
        self.assertEqual(sg_ev["conversations"][1]["value"], "42")  # stripped <think>
        sg_hb = next(r for r in sg_rows if r["metadata"]["harness"] == "pi")
        self.assertIn("function_call", [c["from"] for c in sg_hb["conversations"]])
        self.assertIn("observation", [c["from"] for c in sg_hb["conversations"]])
        self.assertIn("standardize_sharegpt", (out_sg / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
