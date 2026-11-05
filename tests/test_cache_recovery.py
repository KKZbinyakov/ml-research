import builtins
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE = Path(__file__).parents[1] / "app" / "download_model.py"
spec = importlib.util.spec_from_file_location("download_model", MODULE)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class CacheRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parents[1])
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.staging = self.root / "model.staging"
        self.backup = self.root / "model.previous"
        self.env = patch.multiple(loader, MODEL_DIR=self.model, STAGING=self.staging, BACKUP=self.backup,
                                  MODEL_ID="org/model", MODEL_REVISION="a" * 40)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def make_cache(self, path, revision=None):
        path.mkdir(parents=True)
        payloads = {"config.json": b"{}", "tokenizer.json": b"{}", "tokenizer_config.json": b"{}"}
        header = b'{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        header += b" " * (-len(header) % 8)
        payloads["weights.safetensors"] = len(header).to_bytes(8, "little") + header + b"\x00\x00\x80?"
        for name, data in payloads.items():
            (path / name).write_bytes(data)
        files, hashes = loader.snapshot_files(path)
        (path / loader.MARKER).write_text(json.dumps({"model_id": loader.MODEL_ID,
            "revision": revision or loader.MODEL_REVISION, "files": files, "sha256": hashes}))

    def invalidate(self, path):
        (path / loader.MARKER).write_text("{}")

    def run_offline(self):
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith("huggingface_hub"):
                raise AssertionError("recovery attempted Hugging Face import")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded), patch.dict(os.environ, {"MODEL_REVISION": "a" * 40}):
            self.assertEqual(loader.main(), 0)

    def test_staging_copy_recovers_without_hugging_face(self):
        self.make_cache(self.staging)
        self.run_offline()
        self.assertTrue(loader.valid(self.model))

    def test_backup_recovers_after_target_rename(self):
        self.make_cache(self.backup)
        self.run_offline()
        self.assertTrue(loader.valid(self.model))

    def test_staging_recovery_preserves_backup_if_promotion_fails(self):
        self.make_cache(self.staging)
        self.make_cache(self.backup)
        real_replace = Path.replace
        def interrupted(source, target):
            result = real_replace(source, target)
            if source == self.staging and target == self.model:
                raise OSError("simulated interruption")
            return result
        with patch.object(Path, "replace", interrupted):
            with self.assertRaises(OSError):
                loader.main()
        self.assertTrue(loader.valid(self.backup))
        self.run_offline()
        self.assertTrue(loader.valid(self.model))

    def test_interrupted_target_rename_recovers_from_staging(self):
        self.make_cache(self.model)
        self.invalidate(self.model)
        self.make_cache(self.staging)
        real_replace = Path.replace
        def interrupted(source, target):
            result = real_replace(source, target)
            if source == self.model and target == self.backup:
                raise OSError("simulated interruption")
            return result
        with patch.object(Path, "replace", interrupted):
            with self.assertRaises(OSError):
                loader.main()
        self.assertTrue(self.backup.exists())
        self.run_offline()
        self.assertTrue(loader.valid(self.model))

    def test_invalid_target_yields_to_valid_staging_and_backup(self):
        self.make_cache(self.model)
        self.invalidate(self.model)
        self.make_cache(self.staging)
        self.make_cache(self.backup)
        self.run_offline()
        self.assertTrue(loader.valid(self.model))
        self.assertFalse(self.backup.exists())

    def test_same_size_corruption_is_rejected(self):
        self.make_cache(self.model)
        weights = self.model / "weights.safetensors"
        data = weights.read_bytes()
        weights.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        self.assertEqual(weights.stat().st_size, len(data))
        self.assertFalse(loader.valid(self.model))

    def test_wrong_revision_candidates_are_not_recovered(self):
        self.make_cache(self.staging, "b" * 40)
        self.assertFalse(loader.valid(self.staging))
        self.make_cache(self.backup, "b" * 40)
        self.assertFalse(loader.valid(self.backup))

    def test_legacy_marker_without_hashes_is_invalid(self):
        self.make_cache(self.model)
        marker = json.loads((self.model / loader.MARKER).read_text())
        marker.pop("sha256")
        (self.model / loader.MARKER).write_text(json.dumps(marker))
        self.assertFalse(loader.valid(self.model))

    def test_non_object_markers_are_cache_misses(self):
        for contents in ("[]", "null"):
            with self.subTest(contents=contents):
                self.make_cache(self.model)
                (self.model / loader.MARKER).write_text(contents)
                self.assertFalse(loader.valid(self.model))
                for child in self.model.iterdir():
                    child.unlink()
                self.model.rmdir()

    def test_symlink_cache_root_is_rejected(self):
        real = self.root / "real-model"
        self.make_cache(real)
        try:
            self.model.symlink_to(real, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc.__class__.__name__}")
        self.assertFalse(loader.valid(self.model))


if __name__ == "__main__":
    unittest.main()
