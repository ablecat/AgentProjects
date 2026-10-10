from __future__ import annotations

from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
import json
import shutil
import subprocess
import threading
from typing import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import repo_agent.web as web_module
from repo_agent.models import RunResult, ToolResult
from repo_agent.persistence import EventPage
from repo_agent.run_models import ChangePlan, CheckSummary, RunRecord, utc_now
from repo_agent.sandbox import SandboxError
from repo_agent.service import ServiceSnapshot
from repo_agent.web import HOST, RepoAgentHTTPServer


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def committed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "web-source"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Web Test")
    _git(repo, "config", "user.email", "web@example.invalid")
    (repo / "README.md").write_text("# Fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo.resolve()


@contextmanager
def running_server(
    repo: Path, run_function, *, durable_service=None
) -> Iterator[tuple[str, RepoAgentHTTPServer]]:
    server = RepoAgentHTTPServer(
        (HOST, 0),
        repo,
        run_function=run_function,
        durable_service=durable_service,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://{HOST}:{server.server_address[1]}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def fake_result(**kwargs) -> RunResult:
    return RunResult(
        task=kwargs["task"],
        status="completed",
        answer="Inspection complete <script>alert(1)</script>",
        steps=2,
        tool_results=(
            ToolResult(
                "status-1",
                "git_status",
                True,
                "## HEAD (no branch)\n",
                exit_code=0,
            ),
        ),
    )


def request_json(url: str, payload: object, **headers: str):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8")), exc.headers
    with response:
        return response.status, json.loads(response.read().decode("utf-8")), response.headers


def test_static_workspace_is_packaged_and_hardened(committed_repo: Path) -> None:
    with running_server(committed_repo, fake_result) as (base_url, _):
        request = Request(f"{base_url}/")
        with urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
            assert response.status == HTTPStatus.OK
            assert "Repo Maintainer Agent" in body
            assert "v0.1.0 / Release candidate" in body
            assert "Check execution" in body
            assert "Recent runs" in body
            assert "Content-Security-Policy" in response.headers
            assert response.headers["X-Frame-Options"] == "DENY"

        with urlopen(f"{base_url}/app.js", timeout=5) as response:
            javascript = response.read().decode("utf-8")
            assert "textContent" in javascript
            assert "innerHTML" not in javascript
            assert "check.log_artifact_href" in javascript
            assert "encodeURIComponent(check.log_artifact)" not in javascript
            assert "new window.EventSource" in javascript
            assert "Polling fallback" in javascript
            assert "currentRunIsActive(runId)" in javascript
            assert "schedulePoll(currentResult)" in javascript
            assert "childElementCount > 200" in javascript
            assert "usage.total_tokens" in javascript

        with urlopen(f"{base_url}/", timeout=5) as response:
            body = response.read().decode("utf-8")
            assert 'id="activity-list"' in body
            assert 'id="activity-state"' in body


def test_polling_fallback_reschedules_after_a_transient_fetch_failure() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser behavior regression")
    javascript = (
        Path(web_module.__file__).with_name("web_static") / "app.js"
    ).resolve()
    harness = r"""
const fs = require("fs");
const vm = require("vm");

function makeElement() {
  return {
    hidden: false,
    disabled: false,
    textContent: "",
    dataset: {},
    classList: { add() {}, remove() {} },
    addEventListener() {},
    querySelectorAll() { return []; },
    replaceChildren() { this.childElementCount = 0; },
    append() { this.childElementCount += 1; },
    getAttribute() { return "true"; },
    setAttribute() {},
    requestSubmit() {},
    click() {},
    remove() {},
    childElementCount: 0,
    firstElementChild: { remove() {} },
    scrollTop: 0,
    scrollHeight: 0,
  };
}

const cachedElements = new Map();
const document = {
  querySelector(selector) {
    if (!cachedElements.has(selector)) cachedElements.set(selector, makeElement());
    return cachedElements.get(selector);
  },
  querySelectorAll() { return []; },
  createElement() { return makeElement(); },
  body: makeElement(),
};
const timers = [];
const window = {
  EventSource: undefined,
  addEventListener() {},
  clearTimeout() {},
  setTimeout(callback, delay) {
    timers.push({ callback, delay });
    return timers.length;
  },
};
const sandbox = {
  Blob,
  URL,
  console,
  document,
  fetch: async () => { throw new Error("temporary outage"); },
  navigator: { clipboard: { writeText: async () => {} } },
  window,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);

(async () => {
  await vm.runInContext(
    'currentRunId = "run-1"; currentResult = {run_id: "run-1", status: "running"}; pollAgentRun("run-1")',
    sandbox,
  );
  if (timers.length !== 1 || timers[0].delay !== 900) {
    throw new Error(`expected one 900ms retry timer, got ${JSON.stringify(timers)}`);
  }
})().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        (node, "-e", harness, str(javascript)),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


class FakeDurableService:
    def __init__(self, repo: Path, artifacts: Path) -> None:
        self.repo = repo
        self.artifacts = artifacts
        self.artifacts.mkdir()
        (self.artifacts / "checks").mkdir()
        (self.artifacts / "patch.diff").write_text("diff --git a/a b/a\n", encoding="utf-8")
        (self.artifacts / "report.md").write_text("# Report\n", encoding="utf-8")
        (self.artifacts / "run.json").write_text("{}\n", encoding="utf-8")
        (self.artifacts / "trace.jsonl").write_text("{}\n", encoding="utf-8")
        (self.artifacts / "checks" / "verify-1.log").write_bytes(b"passed\n")
        self.record: RunRecord | None = None
        self.events: tuple[dict[str, object], ...] = (
            {
                "sequence": 1,
                "event": "run_created",
                "timestamp": "2026-09-11T00:00:00.000Z",
            },
            {
                "sequence": 2,
                "event": "workflow_stopped",
                "node": "approval",
                "timestamp": "2026-09-11T00:00:01.000Z",
                "detail": {"authorization": "Bearer must-not-stream"},
            },
        )

    @property
    def queue_depth(self) -> int:
        return 0

    def ready(self) -> bool:
        return True

    def scheduler_snapshot(self) -> ServiceSnapshot:
        queued = (
            (self.record.run_id,)
            if self.record is not None and self.record.status == "queued"
            else ()
        )
        active = (
            (self.record.run_id,)
            if self.record is not None and self.record.status in {"planning", "running"}
            else ()
        )
        return ServiceSnapshot(
            ready=True,
            queue_depth=len(queued),
            queue_capacity=4,
            active_count=len(active),
            max_concurrency=1,
            queued_run_ids=queued,
            active_run_ids=active,
        )

    def queue_position(self, run_id: str) -> int | None:
        queued = self.scheduler_snapshot().queued_run_ids
        return queued.index(run_id) + 1 if run_id in queued else None

    def list_runs(self) -> tuple[RunRecord, ...]:
        return () if self.record is None else (self.record,)

    def create_run(self, **kwargs) -> RunRecord:
        now = utc_now()
        self.record = RunRecord(
            run_id="a" * 32,
            repo_path=str(kwargs["repo_path"]),
            task=kwargs["task"],
            base_ref=kwargs["base_ref"],
            status="awaiting_approval",
            current_node="approval",
            plan=ChangePlan(
                goal="Repair the fixture",
                files=("a",),
                steps=("Edit a",),
                checks=("python-pytest",),
                risks=(),
            ),
            checks=(
                CheckSummary(
                    attempt=0,
                    check_id="python-pytest",
                    status="passed",
                    ok=True,
                    duration_ms=12,
                    log_artifact="checks/verify-1.log",
                ),
            ),
            auto_approve=kwargs["auto_approve"],
            allow_remote_model=kwargs["allow_remote_model"],
            created_at=now,
            updated_at=now,
        )
        return self.record

    def get_run(self, run_id: str) -> RunRecord:
        if self.record is None or self.record.run_id != run_id:
            raise web_module.RunNotFoundError(run_id)
        return self.record

    def decide(self, run_id: str, *, approve: bool, reason=None) -> RunRecord:
        record = self.get_run(run_id)
        self.record = RunRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "status": "queued" if approve else "rejected",
                "approval_reason": reason,
            }
        )
        return self.record

    def resume(self, run_id: str) -> RunRecord:
        return self.get_run(run_id)

    def cancel(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        self.record = RunRecord.model_validate(
            {**record.model_dump(mode="python"), "status": "cancelled"}
        )
        return self.record

    def artifact_path(self, run_id: str, kind: str) -> Path:
        self.get_run(run_id)
        names = {
            "patch": "patch.diff",
            "report": "report.md",
            "result": "run.json",
            "trace": "trace.jsonl",
            "checks": "checks",
        }
        return self.artifacts / names[kind]

    def event_page(
        self, run_id: str, *, after_sequence: int, limit: int = 64
    ) -> EventPage:
        self.get_run(run_id)
        events = tuple(
            event for event in self.events if event["sequence"] > after_sequence
        )[:limit]
        latest = max((int(event["sequence"]) for event in self.events), default=0)
        return EventPage(events, latest)

    def wait_for_events(
        self,
        run_id: str,
        *,
        after_sequence: int,
        timeout: float,
        limit: int = 64,
    ) -> EventPage:
        del timeout
        return self.event_page(
            run_id, after_sequence=after_sequence, limit=limit
        )


def test_durable_workspace_create_list_decide_and_artifacts(
    committed_repo: Path, tmp_path: Path
) -> None:
    service = FakeDurableService(committed_repo, tmp_path / "artifacts")
    with running_server(
        committed_repo, fake_result, durable_service=service
    ) as (base_url, _):
        status, created, _ = request_json(
            f"{base_url}/api/agent/runs",
            {
                "task": "repair fixture",
                "base_ref": "HEAD",
                "auto_approve": False,
                "allow_remote_model": True,
            },
        )
        assert status == HTTPStatus.ACCEPTED
        assert created["status"] == "awaiting_approval"
        assert created["plan"]["goal"] == "Repair the fixture"
        assert created["artifacts"]["patch"].endswith("/artifacts/patch")
        log_href = created["checks"][0]["log_artifact_href"]
        assert log_href.endswith("/artifacts/checks/verify-1.log")

        with urlopen(f"{base_url}{log_href}", timeout=5) as response:
            assert response.read() == b"passed\n"

        with urlopen(f"{base_url}/api/agent/runs", timeout=5) as response:
            listing = json.loads(response.read().decode("utf-8"))
        assert listing["ready"] is True
        assert listing["queue_depth"] == 0
        assert listing["queue_capacity"] == 4
        assert listing["active_count"] == 0
        assert listing["max_concurrency"] == 1
        assert [item["run_id"] for item in listing["runs"]] == ["a" * 32]

        status, decided, _ = request_json(
            f"{base_url}/api/agent/runs/{'a' * 32}/decision",
            {"approve": True, "reason": "reviewed"},
        )
        assert status == HTTPStatus.OK
        assert decided["status"] == "queued"
        assert decided["approval_reason"] == "reviewed"

        with urlopen(
            f"{base_url}/api/agent/runs/{'a' * 32}/artifacts/patch",
            timeout=5,
        ) as response:
            assert response.read().startswith(b"diff --git")
            assert response.headers["Content-Disposition"] == (
                'attachment; filename="patch.diff"'
            )

        with urlopen(
            f"{base_url}/api/agent/runs/{'a' * 32}/artifacts/checks",
            timeout=5,
        ) as response:
            checks = json.loads(response.read().decode("utf-8"))
        assert checks["files"][0]["name"] == "verify-1.log"


def test_durable_workspace_streams_replayable_redacted_activity(
    committed_repo: Path, tmp_path: Path
) -> None:
    service = FakeDurableService(committed_repo, tmp_path / "artifacts")
    service.create_run(
        task="repair fixture",
        base_ref="HEAD",
        auto_approve=False,
        allow_remote_model=False,
        repo_path=committed_repo,
    )
    with running_server(
        committed_repo, fake_result, durable_service=service
    ) as (base_url, _):
        with urlopen(
            f"{base_url}/api/agent/runs/{'a' * 32}/events?after=0",
            timeout=5,
        ) as response:
            body = response.read().decode("utf-8")
            assert response.status == HTTPStatus.OK
            assert response.headers["Content-Type"] == (
                "text/event-stream; charset=utf-8"
            )
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Accel-Buffering"] == "no"
            assert response.headers.get("Content-Length") is None

        assert body.startswith("retry: 1500\nevent: snapshot\n")
        assert "id: 1\nevent: progress\n" in body
        assert "id: 2\nevent: progress\n" in body
        assert "id: 2\nevent: done\n" in body
        assert "workflow_stopped" in body
        assert "must-not-stream" not in body
        assert '"detail"' not in body

        with urlopen(
            f"{base_url}/api/agent/runs/{'a' * 32}/events",
            timeout=5,
        ) as response:
            snapshot = response.read().decode("utf-8")
        assert snapshot.startswith("retry: 1500\nid: 2\nevent: snapshot\n")
        assert "event: progress" not in snapshot

        request = Request(
            f"{base_url}/api/agent/runs/{'a' * 32}/events",
            headers={"Last-Event-ID": "1"},
        )
        with urlopen(request, timeout=5) as response:
            resumed = response.read().decode("utf-8")

        stale_request = Request(
            f"{base_url}/api/agent/runs/{'a' * 32}/events",
            headers={"Last-Event-ID": "9"},
        )
        with urlopen(stale_request, timeout=5) as response:
            reset = response.read().decode("utf-8")

    assert "id: 1\nevent: progress\n" not in resumed
    assert "id: 2\nevent: progress\n" in resumed
    assert reset.startswith("retry: 1500\nid: 2\nevent: snapshot\n")
    assert '"cursor":2,"reset":true' in reset


def test_durable_workspace_does_not_link_unsafe_check_artifact_paths(
    committed_repo: Path, tmp_path: Path
) -> None:
    service = FakeDurableService(committed_repo, tmp_path / "artifacts")
    service.create_run(
        task="repair fixture",
        base_ref="HEAD",
        auto_approve=False,
        allow_remote_model=False,
        repo_path=committed_repo,
    )
    assert service.record is not None
    unsafe = service.record.checks[0].model_copy(
        update={"log_artifact": "checks/../private.log"}
    )
    service.record = service.record.model_copy(update={"checks": (unsafe,)})

    with running_server(
        committed_repo, fake_result, durable_service=service
    ) as (base_url, _):
        with urlopen(
            f"{base_url}/api/agent/runs/{'a' * 32}", timeout=5
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert "log_artifact_href" not in payload["checks"][0]

        with pytest.raises(HTTPError) as rejected:
            urlopen(
                f"{base_url}/api/agent/runs/{'a' * 32}/artifacts/checks/..%2Fprivate.log",
                timeout=5,
            )
        assert rejected.value.code == HTTPStatus.NOT_FOUND


def test_run_api_returns_structured_result_and_fixed_metadata(
    committed_repo: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def recording_run(**kwargs) -> RunResult:
        calls.append(kwargs)
        return fake_result(**kwargs)

    with running_server(committed_repo, recording_run) as (base_url, _):
        status, payload, headers = request_json(
            f"{base_url}/api/runs",
            {
                "task": "  inspect <script>  ",
                "image": "repo-agent-python:0.1",
                "max_steps": 5,
                "timeout_seconds": 12,
                "max_output_bytes": 16384,
            },
        )

    assert status == HTTPStatus.OK
    assert payload["status"] == "completed"
    assert payload["task"] == "inspect <script>"
    assert payload["answer"].endswith("<script>alert(1)</script>")
    assert payload["metadata"]["repository"] == str(committed_repo)
    assert payload["metadata"]["image"] == "repo-agent-python:0.1"
    assert len(payload["metadata"]["head"]) == 12
    assert headers["Access-Control-Allow-Origin"] is None
    assert calls[0]["repo_path"] == committed_repo
    assert calls[0]["task"] == "inspect <script>"
    assert calls[0]["allow_mutations"] is False


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"task": ""}, "non-whitespace"),
        ({"task": "inspect", "repo": "C:/Users"}, "Unexpected field"),
        ({"task": "inspect", "image": "alpine:latest"}, "verified sandbox"),
        ({"task": "inspect", "image": []}, "verified sandbox"),
        ({"task": "inspect", "max_steps": 0}, "max_steps"),
        ({"task": "inspect", "timeout_seconds": float("inf")}, "must be finite"),
    ],
)
def test_run_api_rejects_unsafe_or_invalid_payloads(
    committed_repo: Path, payload: object, message: str
) -> None:
    with running_server(committed_repo, fake_result) as (base_url, _):
        status, body, _ = request_json(f"{base_url}/api/runs", payload)

    assert status == HTTPStatus.BAD_REQUEST
    assert message in body["error"]["message"]


def test_run_api_maps_sandbox_setup_failures_to_503(committed_repo: Path) -> None:
    def failed_run(**kwargs):
        raise SandboxError(f"Docker unavailable for {kwargs['task']}")

    with running_server(committed_repo, failed_run) as (base_url, _):
        status, payload, _ = request_json(
            f"{base_url}/api/runs", {"task": "inspect"}
        )

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert payload["status"] == "setup_error"
    assert payload["steps"] == 0
    assert "Docker unavailable" in payload["error"]


def test_run_api_returns_429_while_an_inspection_is_active(
    committed_repo: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    first_response: list[tuple[int, object, object]] = []

    def blocking_run(**kwargs) -> RunResult:
        entered.set()
        assert release.wait(timeout=5)
        return fake_result(**kwargs)

    with running_server(committed_repo, blocking_run) as (base_url, _):
        first = threading.Thread(
            target=lambda: first_response.append(
                request_json(f"{base_url}/api/runs", {"task": "first"})
            )
        )
        first.start()
        assert entered.wait(timeout=5)

        status, payload, _ = request_json(
            f"{base_url}/api/runs", {"task": "second"}
        )
        release.set()
        first.join(timeout=5)

    assert status == HTTPStatus.TOO_MANY_REQUESTS
    assert payload["error"]["code"] == "busy"
    assert first_response[0][0] == HTTPStatus.OK


def test_untrusted_host_origin_and_static_paths_are_rejected(
    committed_repo: Path,
) -> None:
    with running_server(committed_repo, fake_result) as (base_url, _):
        bad_host = Request(f"{base_url}/api/state", headers={"Host": "evil.test"})
        with pytest.raises(HTTPError) as host_error:
            urlopen(bad_host, timeout=5)
        assert host_error.value.code == HTTPStatus.FORBIDDEN

        status, payload, _ = request_json(
            f"{base_url}/api/runs",
            {"task": "inspect"},
            Origin="https://evil.test",
        )
        assert status == HTTPStatus.FORBIDDEN
        assert payload["error"]["code"] == "forbidden"

        with pytest.raises(HTTPError) as path_error:
            urlopen(f"{base_url}/../pyproject.toml", timeout=5)
        assert path_error.value.code == HTTPStatus.NOT_FOUND


def test_state_api_separates_repository_docker_and_policy(
    committed_repo: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        web_module,
        "_application_state",
        lambda repo: {
            "repository": {"path": str(repo), "head": "abc123", "dirty_count": 2},
            "docker": {"available": True, "images": []},
            "policy": {"network": "none"},
        },
    )
    with running_server(committed_repo, fake_result) as (base_url, _):
        with urlopen(f"{base_url}/api/state", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

    assert payload["repository"]["dirty_count"] == 2
    assert payload["docker"]["available"] is True
    assert payload["policy"]["network"] == "none"


def test_application_state_exposes_day3_check_contract(
    committed_repo: Path, monkeypatch
) -> None:
    def fake_capture(argv: tuple[str, ...], *, timeout: float = 5.0) -> str:
        del timeout
        if argv[:2] == ("docker", "version"):
            return "linux|amd64|28.5.1"
        if argv[:3] == ("docker", "context", "show"):
            return "desktop-linux"
        if argv[:3] == ("docker", "image", "inspect"):
            return "sha256:" + "a" * 64
        raise AssertionError(argv)

    monkeypatch.setattr(web_module, "_capture", fake_capture)
    state = web_module._application_state(committed_repo)

    checks = state["checks"]
    assert checks["bootstrap"] == {
        "authorization": "explicit opt-in",
        "network": "bridge",
    }
    assert checks["verify"] == {"network": "none", "workspace": "fresh copy"}
    assert checks["limits"] == {"phase_seconds": 300, "run_seconds": 1200}
    assert [profile["id"] for profile in checks["profiles"]] == ["python-pytest"]


@pytest.mark.parametrize(
    "disconnect_error",
    (BrokenPipeError(), ConnectionAbortedError(), ConnectionResetError()),
)
def test_response_write_ignores_disconnected_clients(disconnect_error) -> None:
    class DisconnectedWriter:
        def write(self, body: bytes) -> None:
            raise disconnect_error

    handler = object.__new__(web_module.RepoAgentRequestHandler)
    handler.wfile = DisconnectedWriter()
    handler.close_connection = False
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None

    handler._send_bytes(HTTPStatus.OK, b"{}", "application/json")

    assert handler.close_connection is True
