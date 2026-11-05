import json
import sys
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bench"))
import record_environment
import run_experiments
import urllib.request


class ExperimentTests(unittest.TestCase):
    def test_condition_order_reverses_each_pair(self):
        order = run_experiments.condition_order(2)
        self.assertEqual([label for _, label in order[:4]], list(run_experiments.CONDITIONS))
        self.assertEqual([label for _, label in order[4:]], list(reversed(run_experiments.CONDITIONS)))

    def test_config_edit_changes_only_two_numeric_values(self):
        text = "max-num-seqs: 4\nmax-num-batched-tokens: 2048\nenable-chunked-prefill: true\nno-enable-prefix-caching: true\n"
        values = {"max-num-seqs": 8, "max-num-batched-tokens": 2048}
        changed = run_experiments.update_serve_yaml(text, values)
        self.assertEqual(changed, text.replace("max-num-seqs: 4", "max-num-seqs: 8"))
        self.assertEqual(record_environment.parse_yaml_config(changed), {
            "max-num-seqs": "8", "max-num-batched-tokens": "2048",
            "enable-chunked-prefill": "true", "no-enable-prefix-caching": "true"})

    def test_metrics_parser_handles_labels_and_sums_series(self):
        text = "# HELP counter help\nvllm:prefix_cache_hits_total{engine=\"0\"} 0\nvllm:prefix_cache_hits_total{engine=\"1\"} 2\n"
        self.assertEqual(run_experiments.metric_counter(text, "vllm:prefix_cache_hits_total"), 2)
        self.assertIsNone(run_experiments.metric_counter("other 3", "vllm:prefix_cache_hits_total"))

    def test_metrics_request_uses_bearer_auth_without_printing_token(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"metric 0\n"
        opener = mock.MagicMock()
        opener.open.return_value = response
        exp = run_experiments.Experiment("kube", 18001, 60, "test-secret")
        with mock.patch.object(run_experiments.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(exp.metrics(), "metric 0\n")
        request = opener.open.call_args.args[0]
        self.assertIsInstance(request, urllib.request.Request)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertNotIn("test-secret", repr(request.full_url))

    def test_restores_original_even_if_configuration_rollout_raises(self):
        original_text = "max-num-seqs: 4\nmax-num-batched-tokens: 2048\n"
        class FakeExperiment:
            def __init__(self, *_):
                self.text = original_text
                self.sets = []
            def get_config(self):
                return {"data": {"serve.yaml": self.text}}
            def stop_forward(self): pass
            def set_serve_yaml(self, text):
                self.sets.append(text)
                self.text = text
                if len(self.sets) == 1:
                    raise RuntimeError("rollout")

        temp = ROOT / "artifacts/test-experiment-temp"
        temp.mkdir(parents=True, exist_ok=True)
        try:
            fake = FakeExperiment()
            args = SimpleNamespace(kubeconfig="dummy", port=18001, rollout_timeout=60, token_file=None,
                                   output_dir=temp, requests=1, repeats=1)
            with mock.patch.object(run_experiments, "Experiment", return_value=fake), \
                 mock.patch.object(run_experiments, "token_from_file", return_value="secret"), \
                 mock.patch.object(run_experiments.record_environment, "record", return_value={}):
                with self.assertRaises(RuntimeError):
                    run_experiments.run(args)
            self.assertEqual(fake.text, original_text)
            progress = json.loads((Path(temp) / "manifest.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertTrue(progress["restored_original_config"])
            self.assertEqual(len(fake.sets), 2)  # rollout is forced even when values already match
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_restore_failure_never_marks_manifest_complete(self):
        original_text = "max-num-seqs: 4\nmax-num-batched-tokens: 2048\n"
        class FakeExperiment:
            def __init__(self, *_): self.calls = 0
            def get_config(self): return {"data": {"serve.yaml": original_text}}
            def stop_forward(self): pass
            def set_serve_yaml(self, text):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("condition rollout")
                raise ValueError("restore rollout")
        temp = ROOT / "artifacts/test-experiment-temp"
        temp.mkdir(parents=True, exist_ok=True)
        fake = FakeExperiment()
        args = SimpleNamespace(kubeconfig="dummy", port=18001, rollout_timeout=60, token_file=None,
                               output_dir=temp, requests=1, repeats=1)
        try:
            with mock.patch.object(run_experiments, "Experiment", return_value=fake), \
                 mock.patch.object(run_experiments, "token_from_file", return_value="secret"):
                with self.assertRaises(ValueError):
                    run_experiments.run(args)
            manifest = json.loads((temp / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["phase"], "restoration_failed")
            self.assertFalse(manifest["restored_original_config"])
        finally:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
