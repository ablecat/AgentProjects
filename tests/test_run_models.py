from __future__ import annotations

import pytest
from pydantic import ValidationError

from repo_agent.run_models import ChangePlan


def test_change_plan_is_strict_and_typed() -> None:
    plan = ChangePlan(
        goal="Repair pagination",
        files=("src/pager.py", "tests/test_pager.py"),
        steps=("Reproduce the boundary case", "Apply the smallest repair"),
        checks=("python-pytest",),
        risks=("Preserve empty-page behavior",),
    )

    assert plan.files == ("src/pager.py", "tests/test_pager.py")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChangePlan(
            goal="Repair",
            files=(),
            steps=("step",),
            checks=("pytest",),
            unexpected=True,
        )


@pytest.mark.parametrize(
    "path",
    ["../outside.py", "/absolute.py", r"src\windows.py", ".git/config"],
)
def test_change_plan_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="plan files"):
        ChangePlan(
            goal="Repair",
            files=(path,),
            steps=("step",),
            checks=("pytest",),
        )


def test_change_plan_rejects_case_insensitive_duplicates() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        ChangePlan(
            goal="Repair",
            files=("src/App.py", "SRC/app.py"),
            steps=("step",),
            checks=("pytest",),
        )
