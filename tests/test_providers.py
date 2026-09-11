from repo_agent.models import FinalAnswer, ToolResult
from repo_agent.providers import DemoProvider, Provider


def test_demo_provider_calls_status_map_search_then_finishes() -> None:
    provider = DemoProvider()

    status_call = provider.next_step("inspect", ())
    assert status_call.id == "demo-git-status"
    assert status_call.name == "git_status"
    assert status_call.arguments == {}

    status_result = ToolResult(
        call_id=status_call.id,
        name=status_call.name,
        ok=True,
        output="On branch main",
    )
    map_call = provider.next_step("inspect", (status_result,))
    assert map_call.id == "demo-repo-map"
    assert map_call.name == "repo_map"
    assert map_call.arguments == {"max_files": 200, "include_symbols": True}

    map_result = ToolResult(
        call_id=map_call.id,
        name=map_call.name,
        ok=True,
        output="FILES 3 shown / 3 visible",
    )
    search_call = provider.next_step("inspect", (status_result, map_result))
    assert search_call.id == "demo-search-todos"
    assert search_call.name == "search"
    assert search_call.arguments == {"pattern": "TODO|FIXME"}

    search_result = ToolResult(
        call_id=search_call.id,
        name=search_call.name,
        ok=True,
        output="src/app.py:4: TODO",
    )
    answer = provider.next_step(
        "inspect", (status_result, map_result, search_result)
    )

    assert isinstance(answer, FinalAnswer)
    assert answer.content == (
        "Demo inspection complete. Git status: On branch main. "
        "Repository map: FILES 3 shown / 3 visible. "
        "TODO/FIXME search: src/app.py:4: TODO."
    )


def test_demo_provider_summarizes_failures_without_raising() -> None:
    provider = DemoProvider()
    failed_status = ToolResult(
        call_id="status",
        name="git_status",
        ok=False,
        output="",
        error="not a git repository",
    )
    failed_search = ToolResult(
        call_id="search",
        name="search",
        ok=False,
        output="",
        error="unknown tool",
    )
    failed_map = ToolResult(
        call_id="map",
        name="repo_map",
        ok=False,
        output="",
        error="map failed",
    )

    answer = provider.next_step(
        "inspect", (failed_status, failed_map, failed_search)
    )

    assert isinstance(answer, FinalAnswer)
    assert answer.content.startswith("Demo inspection finished with tool errors.")
    assert "Git status: not a git repository" in answer.content
    assert "Repository map: map failed" in answer.content
    assert "TODO/FIXME search: unknown tool" in answer.content


def test_demo_provider_satisfies_provider_protocol() -> None:
    assert isinstance(DemoProvider(), Provider)
