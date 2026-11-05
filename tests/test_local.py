from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import local


class LocalToolTests(unittest.TestCase):
    def test_cluster_scoped_paths_and_name_validation(self):
        self.assertEqual(local.kubeconfig_path("ml-research"), local.ROOT / ".tools" / "kubeconfig")
        self.assertEqual(local.kubeconfig_path("ml-research-acceptance"),
                         local.ROOT / ".tools" / "kubeconfig-ml-research-acceptance")
        self.assertEqual(local.token_path("ml-research-acceptance"),
                         local.PRIVATE / "api-token-ml-research-acceptance")
        with self.assertRaises(ValueError):
            local.validate_cluster("unrelated")

    def test_blank_token_rejected(self):
        with tempfile.TemporaryDirectory(dir=local.ROOT) as directory:
            token_file = Path(directory) / "token"
            token_file.write_text(" \n\t")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                local.read_token(token_file)

    def test_stale_path_binary_falls_back_to_verified_cached_pinned_binary(self):
        with tempfile.TemporaryDirectory(dir=local.ROOT) as directory:
            root = Path(directory)
            cached = root / ".tools" / "kind.exe"
            cached.parent.mkdir()
            cached.write_bytes(b"kind")
            with (mock.patch.object(local, "ROOT", root),
                  mock.patch.object(local.platform, "system", return_value="Windows"),
                  mock.patch.object(local.platform, "machine", return_value="AMD64"),
                  mock.patch.object(local.shutil, "which", return_value="C:/old/kind.exe"),
                  mock.patch.object(local, "_kind_version", side_effect=["v0.31.0", local.KIND_VERSION])):
                self.assertEqual(local.kind_binary(), cached)

    def test_download_rejects_bad_checksum_without_replacing_cached_file(self):
        with tempfile.TemporaryDirectory(dir=local.ROOT) as directory:
            root = Path(directory)
            destination = root / ".tools" / "kind.exe"
            destination.parent.mkdir()
            destination.write_bytes(b"old")
            responses = [mock.MagicMock(), mock.MagicMock()]
            responses[0].__enter__.return_value.read.return_value = b"new"
            responses[1].__enter__.return_value.read.return_value = b"0" * 64 + b"  kind\n"
            with (mock.patch.object(local, "ROOT", root),
                  mock.patch.object(local.platform, "system", return_value="Windows"),
                  mock.patch.object(local.platform, "machine", return_value="AMD64"),
                  mock.patch.object(local.shutil, "which", return_value=None),
                  mock.patch.object(local, "_kind_version", return_value=None),
                  mock.patch.object(local.urllib.request, "urlopen", side_effect=responses)):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    local.kind_binary()
            self.assertEqual(destination.read_bytes(), b"old")

    def test_down_deletes_only_selected_kind_cluster_and_kubeconfig(self):
        with tempfile.TemporaryDirectory(dir=local.ROOT) as directory:
            root = Path(directory)
            kubeconfig = root / ".tools" / "kubeconfig-ml-research-acceptance"
            kubeconfig.parent.mkdir()
            kubeconfig.write_text("config")
            with mock.patch.object(local, "ROOT", root), \
                 mock.patch.object(local, "kind_binary", return_value="kind"), \
                 mock.patch.object(local, "run") as run:
                local.down("ml-research-acceptance", kubeconfig)
            self.assertEqual(run.call_args.args[0], ["kind", "delete", "cluster", "--name",
                                                     "ml-research-acceptance"])
            self.assertFalse(kubeconfig.exists())

    def test_down_preserves_external_kubeconfig(self):
        with tempfile.TemporaryDirectory(dir=local.ROOT) as directory:
            root = Path(directory)
            kubeconfig = root / "shared-kubeconfig"
            kubeconfig.write_text("config")
            with mock.patch.object(local, "kind_binary", return_value="kind"), \
                 mock.patch.object(local, "run"):
                local.down("ml-research-acceptance", kubeconfig)
            self.assertTrue(kubeconfig.exists())

    def test_up_uses_custom_cluster_for_kind_config_and_image_load(self):
        with tempfile.TemporaryDirectory(dir=local.ROOT) as directory:
            root = Path(directory)
            (root / "kind.yaml").write_text("kind: Cluster\nname: ml-research\n")
            kubeconfig = root / ".tools" / "kubeconfig-ml-research-acceptance"
            calls = []
            generated = []
            def fake_run(args, **kwargs):
                calls.append(args)
                if args[1:3] == ["create", "cluster"]:
                    generated.append(Path(args[args.index("--config") + 1]).read_text())
                return subprocess.CompletedProcess([], 0, "", "")
            with (mock.patch.object(local, "ROOT", root),
                  mock.patch.object(local, "PRIVATE", root / ".secrets"),
                  mock.patch.object(local, "kind_binary", return_value="kind"),
                  mock.patch.object(local, "run", side_effect=fake_run),
                  mock.patch.object(local, "kube")):
                local.up("ml-research-acceptance", kubeconfig)
            load = next(args for args in calls if len(args) > 1 and args[1:3] == ["load", "docker-image"])
            self.assertIn("name: ml-research-acceptance", generated[0])
            self.assertEqual(load[load.index("--name") + 1], "ml-research-acceptance")

    def test_down_preserves_external_target_when_managed_config_is_symlink(self):
        with tempfile.TemporaryDirectory(dir=local.ROOT) as directory:
            root = Path(directory)
            shared = root / "shared-kubeconfig"
            shared.write_text("config")
            managed = root / ".tools/kubeconfig"
            managed.parent.mkdir()
            try:
                managed.symlink_to(shared)
            except OSError:
                self.skipTest("symlinks unavailable on this platform")
            with mock.patch.object(local, "ROOT", root), \
                 mock.patch.object(local, "kind_binary", return_value="kind"), \
                 mock.patch.object(local, "run"):
                local.down("ml-research", shared)
            self.assertEqual(shared.read_text(), "config")

    def test_cli_accepts_cluster_before_or_after_action(self):
        for argv in (["--cluster", "ml-research-acceptance", "status"],
                     ["status", "--cluster", "ml-research-acceptance"]):
            with mock.patch.object(local.sys, "argv", ["local.py", *argv]), \
                 mock.patch.object(local, "kube") as kube:
                local.main()
            self.assertEqual(kube.call_args.args[0],
                             local.kubeconfig_path("ml-research-acceptance"))

    def test_benchmark_and_verify_receive_port_as_base_url_before_overrides(self):
        for action, script in (("benchmark", "bench/run.py"), ("verify", "tools/verify_runtime.py")):
            with mock.patch.object(local.sys, "argv", ["local.py", action, "--port", "19000",
                                                        "--base-url", "http://custom:9000"]), \
                 mock.patch.object(local, "client_env", return_value={}), \
                 mock.patch.object(local, "run") as run:
                local.main()
            command = run.call_args.args[0]
            self.assertEqual(command[1:5], [script, "--base-url", "http://127.0.0.1:19000",
                                            "--base-url"])
            self.assertEqual(command[5], "http://custom:9000")


if __name__ == "__main__":
    unittest.main()
