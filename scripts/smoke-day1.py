"""Run the Day 1 read-only tools against active local sandbox images."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from repo_agent.cli import main as cli_main  # noqa: E402
from repo_agent.models import ToolCall  # noqa: E402
from repo_agent.sandbox import DockerSandbox  # noqa: E402


PYTHON_IMAGE = "repo-agent-python:0.1"
MAVEN_IMAGE = "repo-agent-maven:0.1"
EXTRA_CALLS = (
    ToolCall("smoke-diff", "git_diff", {}),
    ToolCall("smoke-files", "list_files", {}),
    ToolCall(
        "smoke-read",
        "read_file",
        {"path": "README.md", "start_line": 1, "end_line": 5},
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summaries: list[dict[str, object]] = []
    failed = False

    for image in _selected_images(args.include_legacy_maven):
        stdout = StringIO()
        with redirect_stdout(stdout):
            cli_exit_code = cli_main(
                [
                    "run",
                    "--repo",
                    str(PROJECT_ROOT),
                    "--task",
                    "Inspect repository status and TODO markers",
                    "--provider",
                    "demo",
                    "--image",
                    image,
                    "--format",
                    "json",
                ]
            )
        agent_result = json.loads(stdout.getvalue())

        with DockerSandbox(PROJECT_ROOT, image=image) as sandbox:
            extra_results = [sandbox.execute(call) for call in EXTRA_CALLS]

        agent_tools = agent_result.get("tool_results", [])
        failed = (
            failed
            or cli_exit_code != 0
            or agent_result.get("status") != "completed"
            or any(not result.ok for result in extra_results)
        )
        summaries.append(
            {
                "image": image,
                "agent": {
                    "exit_code": cli_exit_code,
                    "status": agent_result.get("status"),
                    "steps": agent_result.get("steps"),
                },
                "tools": [
                    {
                        "name": result["name"],
                        "ok": result["ok"],
                        "exit_code": result["exit_code"],
                        "truncated": result["truncated"],
                        "output_bytes": len(result["output"].encode("utf-8")),
                        "error": result["error"],
                    }
                    for result in agent_tools
                ]
                + [
                    {
                        "name": result.name,
                        "ok": result.ok,
                        "exit_code": result.exit_code,
                        "truncated": result.truncated,
                        "output_bytes": len(result.output.encode("utf-8")),
                        "error": result.error,
                    }
                    for result in extra_results
                ],
            }
        )

    print(
        json.dumps(
            {"status": "failed" if failed else "passed", "images": summaries},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Day 1 smoke checks against active sandbox images."
    )
    parser.add_argument(
        "--include-legacy-maven",
        action="store_true",
        help="also check the legacy Maven sandbox image",
    )
    return parser.parse_args(argv)


def _selected_images(include_legacy_maven: bool) -> tuple[str, ...]:
    images = [PYTHON_IMAGE]
    if include_legacy_maven:
        images.append(MAVEN_IMAGE)
    return tuple(images)


if __name__ == "__main__":
    raise SystemExit(main())
