"""Inventory distributable application inputs and verify imports, syntax and notebook sources."""

import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DIRECTORIES = ("src", "config", "scripts", "tests", "assets", ".streamlit", "docs", "data/sample", "notebooks")
TOP_LEVEL = ("app.py", "README.md", "requirements.txt", "requirements-dev.txt", "constraints.txt", "pyproject.toml", ".env.example", ".gitignore", ".gitattributes")
EXCLUDED = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache"}
SECRET_PATTERN = re.compile(rb"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])")


def inventory() -> list[Path]:
    paths = {ROOT / name for name in TOP_LEVEL}
    for name in DIRECTORIES:
        paths.update(path for path in (ROOT / name).rglob("*") if path.is_file())
    return sorted(path for path in paths if not set(path.relative_to(ROOT).parts) & EXCLUDED
                  and path.suffix not in {".pyc", ".pyo"}
                  and (not path.name.startswith(".env") or path.name == ".env.example")
                  and path.name != "secrets.toml")


def import_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    visited, active, cycles = set(), [], []

    def visit(module: str) -> None:
        if module in active:
            cycle = active[active.index(module):] + [module]
            if cycle not in cycles:
                cycles.append(cycle)
            return
        if module in visited:
            return
        active.append(module)
        for dependency in sorted(graph.get(module, ())):
            visit(dependency)
        active.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)
    return cycles


def test_evidence() -> tuple[list[dict], datetime | None]:
    """Measure executed JUnit runs and retain the names of reproduced failed assertions."""
    qa = ROOT / "outputs/qa"
    paths = sorted(set(qa.glob("audit-*.xml")) | {qa / "pytest-results.xml", qa / "clean-final-environment-tests.xml"})
    records, baseline = [], None
    for path in paths:
        if not path.is_file():
            continue
        root = ET.parse(path).getroot()
        suites = list(root.iter("testsuite"))
        record = {"evidence": path.relative_to(ROOT).as_posix()}
        record.update({key: sum(int(suite.get(key, 0)) for suite in suites) for key in ["tests", "failures", "errors", "skipped"]})
        record["seconds"] = sum(float(suite.get("time", 0)) for suite in suites)
        record["failed_cases"] = [f"{case.get('classname')}.{case.get('name')}" for case in root.iter("testcase") if case.find("failure") is not None or case.find("error") is not None]
        records.append(record)
        if path.name == "audit-baseline.xml" and suites:
            baseline = datetime.fromisoformat(suites[0].get("timestamp"))
            baseline = baseline.astimezone(timezone.utc)
    return records, baseline


def issue_evidence() -> dict:
    """Count the uniquely identified issue rows in the recorded final audit ledger."""
    path = ROOT / "docs/FINAL_PROJECT_AUDIT.md"
    rows = re.findall(r"^\| (SV-\d{3}) \| (Critical|High|Medium|Low) \|.*\| (Fixed|Open) \|$", path.read_text(encoding="utf-8") if path.is_file() else "", re.MULTILINE)
    if len(rows) != len({row[0] for row in rows}):
        raise ValueError("Duplicate identifiers in the final audit issue ledger.")
    return {
        "identified": len(rows), "fixed": sum(status == "Fixed" for _, _, status in rows),
        "severity": {severity: sum(level == severity for _, level, _ in rows) for severity in ["Critical", "High", "Medium", "Low"]},
        "remaining_critical_high": sum(level in {"Critical", "High"} and status != "Fixed" for _, level, status in rows),
    }


def audit() -> dict:
    files, graph, modules = [], {}, []
    notebook_sources = []
    for path in inventory():
        content = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        if SECRET_PATTERN.search(content):
            raise ValueError(f"Credential-shaped content in distributable file: {relative}")
        record = {"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
        if path.suffix == ".py":
            text = content.decode("utf-8-sig")
            tree = ast.parse(text, filename=relative)
            compile(tree, relative, "exec")
            record.update({"lines": len(text.splitlines()), "inspection": "Python AST and compilation"})
            module = relative.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
                elif isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
            graph[module] = {name for name in imports if name.startswith(("src.", "config.", "scripts."))}
            if relative.startswith(("src/", "config/", "scripts/")):
                modules.append(module)
        elif path.suffix == ".ipynb":
            notebook = nbformat.reads(content.decode("utf-8"), as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    compile(cell.source, f"{relative}:{cell.id}", "exec")
                    notebook_sources.append({"id": cell.id, "source": cell.source})
            record["inspection"] = "Notebook schema and every code-cell source compiled"
        elif path.suffix in {".json", ".toml"}:
            if path.suffix == ".json":
                json.loads(content)
            else:
                import tomllib

                tomllib.loads(content.decode("utf-8"))
            record["inspection"] = "Configuration/metadata parsed"
        else:
            record["inspection"] = "Content checksum and credential scan; human/source or artifact inspection recorded in final audit"
        files.append(record)
    imported = []
    for module in sorted(set(modules)):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(module)
        imported.append(module)
    cycles = import_cycles(graph)
    if cycles:
        raise ValueError(f"Circular project imports: {cycles}")
    runs, baseline = test_evidence()
    updated = [item["path"] for item in files if baseline is not None and datetime.fromisoformat(item["modified_at"]) >= baseline and not item["path"].endswith(".png")]
    return {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "python": sys.version.split()[0],
        "scope": "Distributable application/configuration/tooling/tests/documentation/notebook/samples/screenshots; runtime output and private source excluded",
        "inspected_files": len(files), "python_files": sum(item["path"].endswith(".py") for item in files),
        "imported_modules": imported, "circular_imports": cycles,
        "notebook_code_cells": len(notebook_sources), "notebook_sources": notebook_sources,
        "credential_scan": "passed", "files": files,
        "baseline_started_at": baseline.isoformat() if baseline is not None else None,
        "audit_updated_paths": updated, "audit_updated_files": len(updated),
        "test_runs": runs,
        "failed_regression_functions": sorted({case.split("[", 1)[0] for run in runs for case in run["failed_cases"]}),
        "issue_counts": issue_evidence(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qa/audit-inventory.json")
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"files", "notebook_sources", "imported_modules"}}, indent=2))


if __name__ == "__main__":
    main()
