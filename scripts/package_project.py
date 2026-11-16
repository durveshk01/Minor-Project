"""Create a portable project ZIP with sources, documentation and reproducible real results."""

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_FILES = (
    "app.py", "README.md", "requirements.txt", "requirements-dev.txt", "constraints.txt",
    "pyproject.toml", ".env.example", ".gitignore", ".gitattributes", ".streamlit/config.toml",
)
DIRECTORIES = ("src", "config", "assets", "scripts", "tests", "docs", "notebooks", "data/sample", "outputs/experiments/uci_daily")
MARKERS = ("data/source/.gitkeep", "data/uploads/.gitkeep", "data/processed/.gitkeep", "models/.gitkeep", "outputs/.gitkeep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/salesvision_project.zip")
    args = parser.parse_args()
    sources = []
    for relative in TOP_LEVEL_FILES + MARKERS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required project file is missing: {relative}")
        sources.append(path)
    for relative in DIRECTORIES:
        directory = ROOT / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Required project directory is missing: {relative}")
        sources.extend(path for path in directory.rglob("*") if path.is_file())
    excluded_parts = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache", ".venv"}
    sources = sorted({
        path for path in sources
        if not set(path.relative_to(ROOT).parts) & excluded_parts
        and (not path.name.startswith(".env") or path.name == ".env.example")
        and path.name != "secrets.toml"
        and path.suffix.lower() not in {".pyc", ".pyo", ".sqlite3", ".db", ".zip", ".log"}
        and not path.name.endswith(("-wal", "-shm"))
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, f"sales-forecasting-system/{path.relative_to(ROOT).as_posix()}")
    print(json.dumps({"archive": str(args.output.resolve()), "files": len(sources), "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
