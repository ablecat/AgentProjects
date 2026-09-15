"""Small cross-platform boundary for commands that must not leave live children."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Sequence


PROCESS_EXIT_TIMEOUT_SECONDS = 2.0
OUTPUT_DRAIN_TIMEOUT_SECONDS = 2.0
_WINDOWS_CREATE_SUSPENDED = 0x00000004


@dataclass(frozen=True)
class CapturedProcess:
    """Bounded byte output and lifecycle result for one owned command."""

    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool


class IsolatedProcess:
    """Own one subprocess and descendants kept inside its OS boundary."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        process_group: int | None,
        windows_job: int | None,
    ) -> None:
        self.process = process
        self._process_group = process_group
        self._windows_job = windows_job
        self._closed = False
        self._lock = threading.Lock()

    def terminate(self) -> None:
        """Terminate the owned process group or Job Object."""

        with self._lock:
            if self._windows_job is not None:
                try:
                    _terminate_windows_job(self._windows_job)
                except OSError:
                    self._kill_process()
                    raise
            elif self._process_group is not None:
                try:
                    _kill_process_group(self._process_group)
                except ProcessLookupError:
                    pass
                except OSError:
                    self._kill_process()
                    raise
            else:
                self._kill_process()

    def close(self) -> None:
        """Kill surviving descendants and release the OS ownership primitive."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            cleanup_error: OSError | None = None
            if self._windows_job is not None:
                handle = self._windows_job
                self._windows_job = None
                try:
                    _terminate_windows_job(handle)
                except OSError as exc:
                    cleanup_error = exc
                    self._kill_process()
                try:
                    _close_windows_handle(handle)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
            elif self._process_group is not None:
                try:
                    _kill_process_group(self._process_group)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    cleanup_error = exc
                    self._kill_process()
            else:
                if self.process.poll() is None:
                    self._kill_process()
            if self.process.poll() is None:
                try:
                    wait_after_termination(self.process)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                raise cleanup_error

    def _kill_process(self) -> None:
        try:
            self.process.kill()
        except (OSError, ProcessLookupError):
            pass


def start_isolated_process(
    argv: Sequence[str], **kwargs: Any
) -> IsolatedProcess:
    """Start a subprocess in an ownership boundary suitable for forced cleanup."""

    options = dict(kwargs)
    if os.name == "nt":
        options["creationflags"] = int(options.get("creationflags", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        ) | _WINDOWS_CREATE_SUSPENDED
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(list(argv), **options)
    process_group = (
        getattr(process, "pid", None) if os.name != "nt" else None
    )
    windows_job: int | None = None
    if os.name == "nt":
        try:
            windows_job = _create_windows_job(process)
            if windows_job is None:
                raise OSError("could not create a Windows process ownership job")
            _resume_windows_process(process.pid)
        except BaseException as exc:
            cleanup_errors: list[BaseException] = []
            if windows_job is not None:
                try:
                    _terminate_windows_job(windows_job)
                except OSError as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
                    try:
                        process.kill()
                    except OSError:
                        pass
                try:
                    _close_windows_handle(windows_job)
                except OSError as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            else:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                wait_after_termination(process)
            except OSError as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
            finally:
                _close_process_streams(process)
            if hasattr(exc, "add_note"):
                for cleanup_error in cleanup_errors:
                    exc.add_note(f"Process setup cleanup also failed: {cleanup_error}")
            raise
    return IsolatedProcess(
        process,
        process_group=process_group if isinstance(process_group, int) else None,
        windows_job=windows_job,
    )


def run_isolated_capture(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    input_bytes: bytes | None = None,
    merge_stderr: bool = False,
    terminate_on_stdout_limit: bool = True,
    terminate_on_stderr_limit: bool = True,
    **kwargs: Any,
) -> CapturedProcess:
    """Run a command with bounded pipes and deterministic ownership cleanup.

    Output limits are enforcement boundaries by default: observing the first
    byte beyond either limit terminates the owned process tree.  A caller may
    opt out for a specific pipe only when truncating that pipe cannot let an
    untrusted or side-effecting command continue unchecked.
    """

    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise ValueError("timeout_seconds must be a supported positive number")
    try:
        timeout = float(timeout_seconds)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "timeout_seconds must be a supported positive number"
        ) from exc
    if not 0 < timeout <= threading.TIMEOUT_MAX:
        raise ValueError("timeout_seconds must be a supported positive number")
    for value, name in (
        (max_stdout_bytes, "max_stdout_bytes"),
        (max_stderr_bytes, "max_stderr_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if input_bytes is not None and not isinstance(input_bytes, bytes):
        raise TypeError("input_bytes must be bytes or None")
    for value, name in (
        (terminate_on_stdout_limit, "terminate_on_stdout_limit"),
        (terminate_on_stderr_limit, "terminate_on_stderr_limit"),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")

    options = dict(kwargs)
    for reserved in ("stdin", "stdout", "stderr"):
        if reserved in options:
            raise TypeError(f"{reserved} is managed by run_isolated_capture")
    options["stdin"] = subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL
    options["stdout"] = subprocess.PIPE
    options["stderr"] = subprocess.STDOUT if merge_stderr else subprocess.PIPE
    options["shell"] = False

    managed = start_isolated_process(argv, **options)
    process = managed.process
    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = False
    stderr_truncated = False
    thread_errors: list[tuple[str, BaseException]] = []
    termination_requested = threading.Event()
    termination_request_lock = threading.Lock()
    thread_streams: list[tuple[threading.Thread, Any]] = []
    started_threads: list[tuple[threading.Thread, Any]] = []
    alive_threads: list[str] = []
    timed_out = False

    def request_termination() -> None:
        with termination_request_lock:
            if termination_requested.is_set():
                return
            termination_requested.set()
            managed.terminate()

    def drain(
        label: str,
        stream: Any,
        target: bytearray,
        limit: int,
        terminate_at_limit: bool,
    ) -> None:
        nonlocal stdout_truncated, stderr_truncated
        limit_exceeded = False
        try:
            while True:
                remaining = limit - len(target)
                # Before truncation, consume at most the first byte beyond the
                # boundary.  read1()/os.read() prevents a buffered read from
                # waiting for the entire request while the child is still live.
                read_size = (
                    8192
                    if limit_exceeded
                    else min(8192, max(remaining, 0) + 1)
                )
                chunk = _read_pipe_chunk(stream, read_size)
                if not chunk:
                    break
                if remaining > 0:
                    target.extend(chunk[:remaining])
                truncated = len(chunk) > max(remaining, 0)
                if label == "stdout":
                    stdout_truncated = stdout_truncated or truncated
                else:
                    stderr_truncated = stderr_truncated or truncated
                if truncated:
                    limit_exceeded = True
                    if terminate_at_limit:
                        request_termination()
                        return
        except BaseException as exc:  # pragma: no cover - OS pipe failures
            thread_errors.append((label, exc))
            try:
                request_termination()
            except BaseException as cleanup_exc:
                thread_errors.append((f"{label} cleanup", cleanup_exc))

    def write_input() -> None:
        if stdin_stream is None or input_bytes is None:
            raise RuntimeError("stdin worker started without an input pipe")
        try:
            stdin_stream.write(input_bytes)
            stdin_stream.flush()
        except BrokenPipeError:
            pass
        except OSError as exc:  # pragma: no cover - OS pipe failures
            if process.poll() is None:
                thread_errors.append(("stdin", exc))
                try:
                    request_termination()
                except BaseException as cleanup_exc:
                    thread_errors.append(("stdin cleanup", cleanup_exc))
        finally:
            try:
                stdin_stream.close()
            except OSError:
                pass

    stdin_stream: Any | None = None
    try:
        stdout_stream = process.stdout
        stderr_stream = None if merge_stderr else process.stderr
        stdin_stream = process.stdin if input_bytes is not None else None
        if stdout_stream is None:
            raise OSError("process stdout pipe is unavailable")
        if not merge_stderr and stderr_stream is None:
            raise OSError("process stderr pipe is unavailable")
        if input_bytes is not None and stdin_stream is None:
            raise OSError("process stdin pipe is unavailable")

        thread_streams.append(
            (
                threading.Thread(
                    target=drain,
                    args=(
                        "stdout",
                        stdout_stream,
                        stdout,
                        max_stdout_bytes,
                        terminate_on_stdout_limit,
                    ),
                    daemon=True,
                ),
                stdout_stream,
            )
        )
        if stderr_stream is not None:
            thread_streams.append(
                (
                    threading.Thread(
                        target=drain,
                        args=(
                            "stderr",
                            stderr_stream,
                            stderr,
                            max_stderr_bytes,
                            terminate_on_stderr_limit,
                        ),
                        daemon=True,
                    ),
                    stderr_stream,
                )
            )
        if stdin_stream is not None:
            thread_streams.append(
                (threading.Thread(target=write_input, daemon=True), stdin_stream)
            )

        for thread, stream in thread_streams:
            thread.start()
            started_threads.append((thread, stream))
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            managed.terminate()
            wait_after_termination(process)
    except BaseException as exc:
        cleanup_errors: list[BaseException] = []
        try:
            managed.terminate()
        except BaseException as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
        try:
            wait_after_termination(process)
        except BaseException as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
        if hasattr(exc, "add_note"):
            for cleanup_error in cleanup_errors:
                exc.add_note(f"Process cleanup also failed: {cleanup_error}")
        raise
    finally:
        try:
            managed.close()
        finally:
            drain_deadline = monotonic() + OUTPUT_DRAIN_TIMEOUT_SECONDS
            for thread, stream in started_threads:
                thread.join(timeout=max(0.0, drain_deadline - monotonic()))
                if not thread.is_alive():
                    try:
                        stream.close()
                    except OSError:
                        pass
            started_ids = {id(thread) for thread, _stream in started_threads}
            for thread, stream in thread_streams:
                if id(thread) not in started_ids:
                    try:
                        stream.close()
                    except OSError:
                        pass
            alive_threads = [
                thread.name
                for thread, _stream in started_threads
                if thread.is_alive()
            ]
            if not alive_threads:
                _close_process_streams(process)

    if alive_threads:
        raise OSError("process pipe worker did not stop after ownership cleanup")
    if thread_errors:
        label, error = thread_errors[0]
        raise OSError(f"failed to process {label}: {error}")
    if process.returncode is None:
        raise OSError("process has no exit status after ownership cleanup")
    return CapturedProcess(
        returncode=process.returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
    )


def _read_pipe_chunk(stream: Any, size: int) -> bytes:
    """Read currently available pipe bytes without filling a buffered request."""

    read_once = getattr(stream, "read1", None)
    if callable(read_once):
        chunk = read_once(size)
    else:
        fileno = getattr(stream, "fileno", None)
        if not callable(fileno):
            raise OSError("process pipe does not support single-read semantics")
        chunk = os.read(fileno(), size)
    if not isinstance(chunk, bytes):
        raise OSError("process pipe returned non-byte output")
    return chunk


def wait_after_termination(process: subprocess.Popen[bytes]) -> None:
    """Reap a terminated direct child without creating another unbounded wait."""

    try:
        process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise OSError("process did not exit after forced termination") from exc
    if process.poll() is None:
        raise OSError("process remained alive after forced termination")


def _kill_process_group(process_group: int) -> None:
    kill_group = getattr(os, "killpg", None)
    if not callable(kill_group):
        raise OSError("process groups are unavailable")
    kill_group(process_group, getattr(signal, "SIGKILL", 9))


def _create_windows_job(process: subprocess.Popen[bytes]) -> int | None:
    if os.name != "nt":
        return None
    process_handle = getattr(process, "_handle", None)
    if process_handle is None:
        raise OSError("Windows process handle is unavailable")

    import ctypes
    from ctypes import wintypes

    ctypes_api: Any = ctypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes_api.WinDLL("kernel32", use_last_error=True)
    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    create_job.restype = wintypes.HANDLE
    set_information = kernel32.SetInformationJobObject
    set_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    set_information.restype = wintypes.BOOL
    assign = kernel32.AssignProcessToJobObject
    assign.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    assign.restype = wintypes.BOOL

    job = create_job(None, None)
    if not job:
        raise ctypes_api.WinError(ctypes_api.get_last_error())
    job_value = int(job)
    information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    configured = set_information(
        job,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    if not configured:
        error = ctypes_api.get_last_error()
        _close_windows_handle(job_value)
        raise ctypes_api.WinError(error)
    if not assign(job, wintypes.HANDLE(int(process_handle))):
        error = ctypes_api.get_last_error()
        _close_windows_handle(job_value)
        raise ctypes_api.WinError(error)
    return job_value


def _resume_windows_process(process_id: int) -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    ctypes_api: Any = ctypes

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel32 = ctypes_api.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    create_snapshot.restype = wintypes.HANDLE
    thread_first = kernel32.Thread32First
    thread_first.argtypes = (wintypes.HANDLE, ctypes.POINTER(THREADENTRY32))
    thread_first.restype = wintypes.BOOL
    thread_next = kernel32.Thread32Next
    thread_next.argtypes = (wintypes.HANDLE, ctypes.POINTER(THREADENTRY32))
    thread_next.restype = wintypes.BOOL
    open_thread = kernel32.OpenThread
    open_thread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_thread.restype = wintypes.HANDLE
    resume_thread = kernel32.ResumeThread
    resume_thread.argtypes = (wintypes.HANDLE,)
    resume_thread.restype = wintypes.DWORD

    snapshot = create_snapshot(0x00000004, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid_handle:
        raise ctypes_api.WinError(ctypes_api.get_last_error())

    resumed = 0
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        has_entry = bool(thread_first(snapshot, ctypes.byref(entry)))
        while has_entry:
            if entry.th32OwnerProcessID == process_id:
                thread = open_thread(0x0002, False, entry.th32ThreadID)
                if not thread:
                    raise ctypes_api.WinError(ctypes_api.get_last_error())
                thread_value = int(thread)
                try:
                    previous_count = int(resume_thread(thread))
                    if previous_count == 0xFFFFFFFF:
                        raise ctypes_api.WinError(ctypes_api.get_last_error())
                    resumed += 1
                finally:
                    _close_windows_handle(thread_value)
            entry.dwSize = ctypes.sizeof(entry)
            has_entry = bool(thread_next(snapshot, ctypes.byref(entry)))
    finally:
        _close_windows_handle(int(snapshot))
    if resumed == 0:
        raise OSError("suspended Windows process thread was not found")


def _terminate_windows_job(handle: int) -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    ctypes_api: Any = ctypes

    kernel32 = ctypes_api.WinDLL("kernel32", use_last_error=True)
    terminate = kernel32.TerminateJobObject
    terminate.argtypes = (wintypes.HANDLE, wintypes.UINT)
    terminate.restype = wintypes.BOOL
    if not terminate(wintypes.HANDLE(handle), 1):
        raise ctypes_api.WinError(ctypes_api.get_last_error())


def _close_windows_handle(handle: int) -> None:
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    ctypes_api: Any = ctypes

    kernel32 = ctypes_api.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    if not close_handle(wintypes.HANDLE(handle)):
        raise ctypes_api.WinError(ctypes_api.get_last_error())


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


__all__ = [
    "CapturedProcess",
    "IsolatedProcess",
    "OUTPUT_DRAIN_TIMEOUT_SECONDS",
    "PROCESS_EXIT_TIMEOUT_SECONDS",
    "run_isolated_capture",
    "start_isolated_process",
    "wait_after_termination",
]
