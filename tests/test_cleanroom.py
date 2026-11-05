import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("cleanroom_under_test", ROOT / "tools/verify_cleanroom.py")
cleanroom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanroom)


class CleanroomTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".tools").mkdir(exist_ok=True)

    def test_archive_rejects_traversal_absolute_and_private_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tools") as temp:
            directory = Path(temp)
            self.assertTrue(directory.resolve().is_relative_to((ROOT / ".tools").resolve()))
            for name in ("../escape", "/absolute", "C:/absolute", "ml_research/.secrets/api-token"):
                with self.subTest(name=name):
                    archive = directory / "test.zip"
                    with zipfile.ZipFile(archive, "w") as package:
                        package.writestr(name, "test")
                    with self.assertRaises(ValueError):
                        cleanroom.extract_archive(archive, directory / "unpacked")

    def test_archive_requires_project_entrypoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tools") as temp:
            directory = Path(temp)
            self.assertTrue(directory.resolve().is_relative_to((ROOT / ".tools").resolve()))
            archive = directory / "test.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("ml_research/README.md", "test")
            with self.assertRaisesRegex(ValueError, "submission_project_missing"):
                cleanroom.extract_archive(archive, directory / "unpacked")


if __name__ == "__main__":
    unittest.main()
