from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_ROOT = PROJECT_ROOT / "benchmarks" / "tasks" / "python" / "development"


def test_development_task_lifecycles_and_content_locks() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (sys.executable, str(DEVELOPMENT_ROOT / "validate.py")),
        cwd=PROJECT_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "[ok] development task structure, locks, and lifecycle" in completed.stdout


def test_development_tasks_are_not_formal_evaluation_tasks() -> None:
    formal = json.loads(
        (PROJECT_ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8")
    )
    development = json.loads(
        (DEVELOPMENT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )

    formal_ids = {entry["id"] for entry in formal["tasks"]}
    development_ids = {entry["id"] for entry in development["tasks"]}
    assert len(formal_ids) == 12
    assert len(development_ids) == 4
    assert formal_ids.isdisjoint(development_ids)
    assert development["formal_evaluation"] is False
