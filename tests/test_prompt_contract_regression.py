from __future__ import annotations

from types import SimpleNamespace

from repo_agent import evaluation
from repo_agent import workflow_runtime


IMMUTABLE_EXISTING_TESTS = (
    "Never modify, delete, or rename a test file that existed when the run began."
)
PYTHON_REGRESSION_PATH = "`tests/test_*_regression.py`"
JAVA_REGRESSION_PATH = "`*RegressionTest.java`"


def _assert_regression_test_contract(prompt: str) -> None:
    assert IMMUTABLE_EXISTING_TESTS in prompt
    assert PYTHON_REGRESSION_PATH in prompt
    assert JAVA_REGRESSION_PATH in prompt
    assert "Python tasks must add a new regression test file" in prompt
    assert "Java tasks must add a new regression test file" in prompt


def test_baseline_prompt_has_explicit_regression_test_contract() -> None:
    _assert_regression_test_contract(evaluation._BASELINE_SYSTEM_PROMPT)


def test_every_durable_model_prompt_has_explicit_regression_test_contract() -> None:
    context = SimpleNamespace(
        run=SimpleNamespace(task="Repair the defect", plan=None, checks=())
    )
    prompts = (
        workflow_runtime._planning_system_prompt(),
        workflow_runtime._implementation_system_prompt(),
        workflow_runtime._change_prompt(context, "implement", ""),
        workflow_runtime._review_system_prompt(),
    )

    for prompt in prompts:
        _assert_regression_test_contract(prompt)
