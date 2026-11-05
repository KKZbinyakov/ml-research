"""Package the submission without local credentials, tooling or Python caches."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
output = root / "artifacts/Збиняков К.К. Команда исследования технологий машинного обучения.zip"
excluded = {".git", ".tools", ".secrets", ".venv", "__pycache__"}
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_file() and not excluded.intersection(relative.parts)
                and path.suffix not in {".pyc", ".zip", ".tar", ".log"}):
            archive.write(path, Path("ml_research") / relative)
print("Created submission archive; local tokens and tooling excluded")
