import sys
import tempfile
import unittest
from pathlib import Path

from app import checklist as cl
from app import runner as rn
from app.errors import CommandNotAllowed
from app.ollama_client import ModelInfo


class RunnerBuilderTests(unittest.TestCase):
    def test_build_run_python_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.py"
            f.write_text("print(1)")
            spec = rn.build_run_python_file(f, tmp)
            self.assertEqual(spec.argv[0], sys.executable)
            rn.validate_spec(spec)

    def test_rejects_non_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.txt"
            f.write_text("x")
            with self.assertRaises(CommandNotAllowed):
                rn.build_run_python_file(f, tmp)

    def test_unittest_pattern_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CommandNotAllowed):
                rn.build_run_unittest(tmp, pattern="bad;rm.py")

    def test_validate_rejects_foreign_program(self):
        spec = rn.CommandSpec(["rm", "-rf", "/"], Path.cwd(), "evil")
        with self.assertRaises(CommandNotAllowed):
            rn.validate_spec(spec)


class RunnerExecutionTests(unittest.TestCase):
    def test_python_version_runs(self):
        result = rn.run_spec(rn.build_python_version())
        self.assertTrue(result.ok)
        self.assertIn("Python", result.combined_output)

    def test_unittest_discovery_on_sample_like_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "test_ok.py").write_text(
                "import unittest\n\n"
                "class T(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertEqual(1, 1)\n", encoding="utf-8")
            spec = rn.build_run_unittest(tmp)
            result = rn.run_spec(spec, timeout=60)
            self.assertTrue(result.ok, result.combined_output)
            summary = rn.parse_unittest_output(result.combined_output)
            self.assertTrue(summary.ok)
            self.assertEqual(summary.ran, 1)

    def test_parse_failures(self):
        output = ("Ran 3 tests in 0.01s\n\nFAILED (failures=1, errors=1, skipped=1)")
        summary = rn.parse_unittest_output(output)
        self.assertEqual((summary.ran, summary.failures, summary.errors, summary.skipped),
                         (3, 1, 1, 1))
        self.assertFalse(summary.ok)

    def test_timeout_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            slow = Path(tmp) / "slow.py"
            slow.write_text("import time\ntime.sleep(30)\n")
            spec = rn.build_run_python_file(slow, tmp)
            with self.assertRaises(rn.CommandTimedOut):
                rn.run_spec(spec, timeout=1.0)


class ChecklistTests(unittest.TestCase):
    def test_python_version_ok(self):
        self.assertEqual(cl.evaluate_python_version((3, 11, 0)).status, cl.CheckStatus.OK)
        self.assertEqual(cl.evaluate_python_version((2, 7, 0)).status, cl.CheckStatus.FAIL)

    def test_server_evaluation(self):
        ok = cl.evaluate_server(True, "0.1.0")
        self.assertEqual(ok.status, cl.CheckStatus.OK)
        bad = cl.evaluate_server(False)
        self.assertEqual(bad.status, cl.CheckStatus.FAIL)
        self.assertTrue(bad.advice)

    def test_model_present_and_sibling(self):
        models = [ModelInfo("qwen2.5-coder:7b", size_bytes=1024)]
        hit = cl.evaluate_model("m", "Main", "qwen2.5-coder:7b", models, True)
        self.assertEqual(hit.status, cl.CheckStatus.OK)
        sibling = cl.evaluate_model("m", "Main", "qwen2.5-coder:14b", models, True)
        self.assertEqual(sibling.status, cl.CheckStatus.WARN)
        missing = cl.evaluate_model("m", "Main", "other:1b", models, True)
        self.assertEqual(missing.status, cl.CheckStatus.FAIL)
        optional = cl.evaluate_model("m", "Main", "other:1b", models, False)
        self.assertEqual(optional.status, cl.CheckStatus.WARN)

    def test_disk_and_ram(self):
        enough = cl.evaluate_disk(50 * 1024 ** 3, 100 * 1024 ** 3)
        self.assertEqual(enough.status, cl.CheckStatus.OK)
        low = cl.evaluate_disk(2 * 1024 ** 3, 100 * 1024 ** 3)
        self.assertEqual(low.status, cl.CheckStatus.WARN)
        self.assertEqual(cl.evaluate_ram(None).status, cl.CheckStatus.UNKNOWN)
        self.assertEqual(cl.evaluate_ram(16 * 1024 ** 3).status, cl.CheckStatus.OK)

    def test_report_score_ready_and_steps(self):
        report = cl.ChecklistReport([
            cl.CheckItem("server", "s", cl.CheckStatus.OK, "up"),
            cl.CheckItem("main_model", "m", cl.CheckStatus.OK, "in"),
            cl.CheckItem("disk", "d", cl.CheckStatus.FAIL, "low", "free up space"),
        ])
        self.assertTrue(report.ready)
        self.assertAlmostEqual(report.score, (1 + 1 + 0) / 3)
        steps = cl.next_steps(report)
        self.assertTrue(any("free up space" in s for s in steps))
        self.assertIn("[X]", report.render())


if __name__ == "__main__":
    unittest.main()
