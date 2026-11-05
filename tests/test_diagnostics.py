import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import MagicMock, patch

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import privacy

spec = importlib.util.spec_from_file_location("serve_under_test", APP / "serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class DiagnosticsTests(unittest.TestCase):
    def test_exception_class_and_component_survive_without_private_text(self):
        try:
            raise ValueError("PROMPT_SECRET TOKEN_SECRET GENERATED_SECRET")
        except ValueError:
            record = logging.LogRecord("vllm.v1.engine.core", logging.ERROR, "private/path", 42,
                                       "PROMPT_SECRET %s", ("TOKEN_SECRET",), sys.exc_info())
        encoded = privacy.SafeFormatter().format(record)
        self.assertEqual(json.loads(encoded), {"event": "runtime_log", "level": "ERROR",
            "component": "engine", "stage": "exception", "error_type": "ValueError"})
        self.assertIsNotNone(privacy.safe_output_line(encoded))
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("private", encoded)

    def test_arbitrary_logger_and_exception_names_are_never_forwarded(self):
        private_exception = type("PRIVATE_RESPONSE", (Exception,), {})
        record = logging.LogRecord("PRIVATE_PROMPT", logging.ERROR, "", 0, "PRIVATE_TOKEN", (),
                                   (private_exception, private_exception("PRIVATE_OUTPUT"), None))
        data = json.loads(privacy.SafeFormatter().format(record))
        self.assertEqual((data["component"], data["error_type"]), ("runtime", "OtherError"))
        self.assertNotIn("PRIVATE", json.dumps(data))
        for key in ("component", "stage", "error_type"):
            bad = dict(data, **{key: "PRIVATE_TEXT"})
            self.assertIsNone(privacy.safe_output_line(json.dumps(bad)))


class SupervisorTests(unittest.TestCase):
    def run_supervisor(self, register_error=None, early_signal=False):
        child = MagicMock()
        child.pid = 12345
        child.wait.return_value = 0
        selector = MagicMock()
        selector.get_map.return_value = {}
        if register_error:
            selector.register.side_effect = register_error
        handlers = {}

        def set_handler(signum, handler):
            old = handlers.get(signum, signal.SIG_DFL)
            handlers[signum] = handler
            return old

        def spawn(*args, **kwargs):
            if early_signal:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return child

        output = io.StringIO()
        with patch.dict(os.environ, {"ML_AUDIT_KEY": "unit-key"}), \
                patch.object(serve.subprocess, "Popen", side_effect=spawn), \
                patch.object(serve.selectors, "DefaultSelector", return_value=selector), \
                patch.object(serve.signal, "signal", side_effect=set_handler), \
                patch.object(serve.os, "killpg", create=True) as kill, contextlib.redirect_stdout(output):
            code = serve.supervise([])
        return code, child, selector, kill, output.getvalue(), handlers

    def test_supervisor_cleans_child_after_selector_failure(self):
        code, child, selector, kill, output, handlers = self.run_supervisor(OSError("PRIVATE_PROMPT"))
        self.assertEqual(code, 1)
        self.assertTrue(child.wait.called)
        child.stdout.close.assert_called_once()
        child.stderr.close.assert_called_once()
        selector.close.assert_called_once()
        self.assertIn(unittest.mock.call(child.pid, signal.SIGTERM), kill.call_args_list)
        self.assertIn(unittest.mock.call(child.pid, serve.KILL_SIGNAL), kill.call_args_list)
        records = [json.loads(x) for x in output.splitlines()]
        self.assertEqual(records[-1]["stage"], "supervision")
        self.assertEqual(records[-1]["error_type"], "OSError")
        self.assertNotIn("PRIVATE", output)
        self.assertEqual(handlers[signal.SIGTERM], signal.SIG_DFL)

    def test_signal_during_spawn_is_forwarded_and_handlers_restored(self):
        code, child, selector, kill, output, handlers = self.run_supervisor(early_signal=True)
        self.assertEqual(code, 0)
        self.assertEqual(kill.call_args_list[0], unittest.mock.call(child.pid, signal.SIGTERM))
        self.assertEqual(handlers[signal.SIGTERM], signal.SIG_DFL)
        self.assertTrue(all(privacy.safe_output_line(x) for x in output.splitlines()))


if __name__ == "__main__":
    unittest.main()
