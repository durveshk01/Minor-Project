"""Install pinned requirements into an isolated audit environment and verify real imports/tests."""

import argparse
import json
import os
import subprocess
import time
import venv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=ROOT / "outputs/qa/clean-runtime-audit")
    parser.add_argument("--install-timeout", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qa/clean-environment-results.json")
    args = parser.parse_args()
    qa = ROOT / "outputs/qa"
    qa.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.stem.removesuffix("-environment-results")
    test_evidence = qa / f"{prefix}-environment-tests.xml"
    environment = args.environment.resolve()
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    evidence = {"started_at": datetime.now(timezone.utc).isoformat(), "environment": str(environment), "status": "running", "checks": []}
    started = time.perf_counter()

    def save() -> None:
        evidence["duration_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    def run(name: str, command: list[str], timeout: int) -> bool:
        tick = time.perf_counter()
        log_path = qa / f"{prefix}-{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                code = result.returncode
                status = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                code, status = None, "timed_out"
        evidence["checks"].append({"name": name, "status": status, "returncode": code, "log": log_path.relative_to(ROOT).as_posix(), "seconds": time.perf_counter() - tick})
        save()
        return status == "passed"

    try:
        evidence["created_environment"] = not executable.is_file()
        if not executable.is_file():
            venv.EnvBuilder(with_pip=True).create(environment)
        initial = subprocess.run([str(executable), "-m", "pip", "list", "--format=json"], capture_output=True, text=True, timeout=60, check=True)
        evidence["initial_packages"] = json.loads(initial.stdout)
        save()
        commands = [
            ("install", [str(executable), "-m", "pip", "install", "--disable-pip-version-check", "--prefer-binary", "--verbose", "-r", "requirements-dev.txt"], args.install_timeout),
            ("dependencies", [str(executable), "-m", "pip", "check"], 60),
            ("imports", [str(executable), "-c", "import numpy, scipy, pandas, statsmodels, prophet, streamlit, google.genai, matplotlib, nbclient, ipykernel; print('Native and application dependencies imported successfully')"], 90),
            ("pytest", [str(executable), "-m", "pytest", "-q", "--tb=short", "--basetemp", str(qa / f"{prefix}-tests"), "--junitxml", str(test_evidence)], 300),
        ]
        for name, command, timeout in commands:
            if not run(name, command, timeout):
                evidence["status"] = "blocked" if evidence["checks"][-1]["status"] == "timed_out" else "failed"
                break
        else:
            suites = list(ET.parse(test_evidence).getroot().iter("testsuite"))
            evidence["pytest"] = {attribute: sum(int(suite.get(attribute, 0)) for suite in suites) for attribute in ("tests", "failures", "errors", "skipped")}
            evidence["status"] = "passed"
    except Exception as exc:
        evidence.update({"status": "failed", "failure_type": type(exc).__name__})
    finally:
        save()
        print(json.dumps(evidence, indent=2))
    if evidence["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
