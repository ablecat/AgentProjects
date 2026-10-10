from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke-day1.py"
SPEC = importlib.util.spec_from_file_location("smoke_day1", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_day1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_day1)


@pytest.mark.parametrize(
    ("arguments", "expected_images"),
    [
        ([], [smoke_day1.PYTHON_IMAGE]),
        (
            ["--include-legacy-maven"],
            [smoke_day1.PYTHON_IMAGE, smoke_day1.MAVEN_IMAGE],
        ),
    ],
)
def test_main_selects_python_by_default_and_maven_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    expected_images: list[str],
) -> None:
    invoked_images: list[str] = []

    def fake_cli_main(argv: list[str]) -> int:
        image = argv[argv.index("--image") + 1]
        invoked_images.append(image)
        print(json.dumps({"status": "completed", "steps": 1, "tool_results": []}))
        return 0

    class FakeSandbox:
        def __init__(self, _repo: Path, *, image: str) -> None:
            self.image = image

        def __enter__(self) -> FakeSandbox:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, call: object) -> SimpleNamespace:
            return SimpleNamespace(
                name=getattr(call, "name"),
                ok=True,
                exit_code=0,
                truncated=False,
                output="ok",
                error=None,
            )

    monkeypatch.setattr(smoke_day1, "cli_main", fake_cli_main)
    monkeypatch.setattr(smoke_day1, "DockerSandbox", FakeSandbox)

    assert smoke_day1.main(arguments) == 0

    payload = json.loads(capsys.readouterr().out)
    assert invoked_images == expected_images
    assert [image["image"] for image in payload["images"]] == expected_images
