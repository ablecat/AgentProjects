from __future__ import annotations

import io
import os
import subprocess
import sys
import time

import pytest

import repo_agent.processes as processes


class _StubbornProcess:
    returncode = None

    def __init__(self) -> None:
        self.kill_calls = 0
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        raise subprocess.TimeoutExpired("stubborn", timeout)

    def kill(self) -> None:
        self.kill_calls += 1

    def poll(self):
        return None


def test_wait_after_termination_reports_an_unreaped_process() -> None:
    process = _StubbornProcess()

    with pytest.raises(OSError, match="did not exit"):
        processes.wait_after_termination(process)  # type: ignore[arg-type]

    assert process.wait_calls == 2
    assert process.kill_calls == 1


def test_isolated_capture_rejects_an_unrepresentable_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        processes.run_isolated_capture(
            (sys.executable, "-c", "pass"),
            timeout_seconds=10**1000,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
        )


def test_isolated_capture_bounds_both_pipes_and_writes_input() -> None:
    script = (
        "import sys; "
        "data = sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(data + b'xx'); "
        "sys.stderr.buffer.write(b'error')"
    )

    result = processes.run_isolated_capture(
        (sys.executable, "-c", script),
        timeout_seconds=5,
        max_stdout_bytes=2,
        max_stderr_bytes=3,
        input_bytes=b"abc",
        terminate_on_stdout_limit=False,
        terminate_on_stderr_limit=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"ab"
    assert result.stderr == b"err"
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert result.timed_out is False


@pytest.mark.parametrize(
    ("stream_name", "limit", "payload"),
    (
        ("stdout", 0, b"x"),
        ("stdout", 4, b"xxxxx"),
        ("stderr", 4, b"xxxxx"),
    ),
)
def test_isolated_capture_stops_on_the_first_byte_over_either_pipe_limit(
    stream_name: str, limit: int, payload: bytes
) -> None:
    target = "stdout" if stream_name == "stdout" else "stderr"
    script = (
        "import sys, time; "
        f"sys.{target}.buffer.write({payload!r}); "
        f"sys.{target}.buffer.flush(); "
        "time.sleep(30)"
    )
    started = time.monotonic()

    result = processes.run_isolated_capture(
        (sys.executable, "-c", script),
        timeout_seconds=10,
        max_stdout_bytes=limit if stream_name == "stdout" else 4,
        max_stderr_bytes=limit if stream_name == "stderr" else 4,
    )

    assert result.returncode != 0
    captured = result.stdout if stream_name == "stdout" else result.stderr
    truncated = (
        result.stdout_truncated
        if stream_name == "stdout"
        else result.stderr_truncated
    )
    assert captured == payload[:limit]
    assert truncated is True
    assert result.timed_out is False
    assert time.monotonic() - started < 5


def test_isolated_capture_does_not_truncate_exact_stdout_limit() -> None:
    result = processes.run_isolated_capture(
        (sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'xxxx')"),
        timeout_seconds=5,
        max_stdout_bytes=4,
        max_stderr_bytes=0,
    )

    assert result.returncode == 0
    assert result.stdout == b"xxxx"
    assert result.stdout_truncated is False
    assert result.timed_out is False


def test_isolated_capture_cleans_up_when_a_worker_cannot_be_constructed(
    monkeypatch,
) -> None:
    class PendingProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = io.BytesIO()
            self.stderr = None
            self.returncode: int | None = None
            self.killed = False

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                raise AssertionError("process must be terminated before waiting")
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = PendingProcess()
    managed = processes.IsolatedProcess(
        process, process_group=None, windows_job=None  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        processes, "start_isolated_process", lambda *_args, **_kwargs: managed
    )

    def reject_worker(*_args, **_kwargs):
        raise RuntimeError("thread construction failed")

    monkeypatch.setattr(processes.threading, "Thread", reject_worker)

    with pytest.raises(RuntimeError, match="thread construction failed"):
        processes.run_isolated_capture(
            ("unused",),
            timeout_seconds=1,
            max_stdout_bytes=1,
            max_stderr_bytes=0,
            merge_stderr=True,
        )

    assert process.killed is True
    assert process.stdout.closed is True


def test_isolated_capture_terminates_promptly_after_a_pipe_read_error(
    monkeypatch,
) -> None:
    created: list[processes.IsolatedProcess] = []
    real_start = processes.start_isolated_process

    def recording_start(*args, **kwargs):
        managed = real_start(*args, **kwargs)
        created.append(managed)
        return managed

    def reject_read(_stream, _size):
        raise OSError("pipe read failed")

    monkeypatch.setattr(processes, "start_isolated_process", recording_start)
    monkeypatch.setattr(processes, "_read_pipe_chunk", reject_read)
    started = time.monotonic()

    with pytest.raises(OSError, match="failed to process"):
        processes.run_isolated_capture(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            timeout_seconds=10,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
        )

    assert time.monotonic() - started < 5
    assert len(created) == 1
    process = created[0].process
    assert process.poll() is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed


def test_isolated_capture_cleans_up_when_a_worker_cannot_start(monkeypatch) -> None:
    class PendingProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stdout = io.BytesIO()
            self.stderr = None
            self.returncode: int | None = None
            self.killed = False

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                raise AssertionError("process must be terminated before waiting")
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    class UnstartableThread:
        name = "unstartable"

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def start(self) -> None:
            raise RuntimeError("thread start failed")

        def is_alive(self) -> bool:
            return False

    process = PendingProcess()
    managed = processes.IsolatedProcess(
        process, process_group=None, windows_job=None  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        processes, "start_isolated_process", lambda *_args, **_kwargs: managed
    )
    monkeypatch.setattr(processes.threading, "Thread", UnstartableThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        processes.run_isolated_capture(
            ("unused",),
            timeout_seconds=1,
            max_stdout_bytes=1,
            max_stderr_bytes=0,
            merge_stderr=True,
        )

    assert process.killed is True
    assert process.stdout.closed is True


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_windows_job_setup_failure_reaps_the_suspended_process(monkeypatch) -> None:
    real_popen = subprocess.Popen
    created: list[subprocess.Popen[bytes]] = []

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        created.append(process)
        return process

    def reject_job(_process):
        raise OSError("job setup rejected")

    monkeypatch.setattr(processes.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(processes, "_create_windows_job", reject_job)

    with pytest.raises(OSError, match="job setup rejected"):
        processes.start_isolated_process(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    assert len(created) == 1
    assert created[0].poll() is not None
    assert created[0].stdout is not None and created[0].stdout.closed
