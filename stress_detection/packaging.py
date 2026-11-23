"""Create a portable source archive without raw data, environments, or credentials."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def package_project(output):
    files = [
        ROOT / name
        for name in (
            "app.py",
            "predict.py",
            "README.md",
            "requirements.txt",
            "requirements-notebook.txt",
            ".gitignore",
        )
    ]
    for directory, patterns in {
        "stress_detection": ["*.py"],
        "tests": ["*.py"],
        "scripts": ["*.py"],
        "notebooks": ["*.ipynb"],
        "assets": ["*.json", "*.png"],
        "models": ["wesad_bundle.joblib"],
    }.items():
        for pattern in patterns:
            files.extend((ROOT / directory).glob(pattern))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
    return output
