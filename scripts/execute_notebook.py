"""Execute the experiment notebook with this interpreter, without global kernel changes."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/Sales_Forecasting_Experiments.ipynb")
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}})
    started = time.perf_counter()
    try:
        client.execute()
    except Exception:
        target = ROOT / "outputs/qa/notebook-failed.ipynb"
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
        raise
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError("The executed notebook contains cell errors.")
    nbformat.write(notebook, args.notebook)
    evidence = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "notebook": str(args.notebook.relative_to(ROOT)),
        "interpreter": sys.executable,
        "duration_seconds": time.perf_counter() - started,
        "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "executed_code_cells": sum(cell.cell_type == "code" and cell.execution_count is not None for cell in notebook.cells),
        "errors": len(errors),
        "execution_environment": "Local Python kernel; Colab upload/Drive branches are provided but were not hosted-Colab tested.",
    }
    target = ROOT / "outputs/qa/notebook-execution.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
