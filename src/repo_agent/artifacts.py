"""Run-owned artifact creation, redaction, and safe lookup."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

from .run_models import ARTIFACT_FILENAMES, ArtifactKind, RunRecord, utc_now


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]+=*"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|authorization|password)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


class ArtifactError(RuntimeError):
    """Raised when a run artifact cannot be created or safely resolved."""


def redact_text(value: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Remove configured and high-confidence credential forms from text."""

    result = value
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        result = result.replace(secret, "[REDACTED]")
    result = _SECRET_PATTERNS[0].sub("Bearer [REDACTED]", result)
    result = _SECRET_PATTERNS[1].sub("[REDACTED]", result)
    result = _SECRET_PATTERNS[2].sub(r"\1\2[REDACTED]", result)
    result = _SECRET_PATTERNS[3].sub("[REDACTED]", result)
    return result


def redact_value(value: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    if isinstance(value, list):
        return [redact_value(item, secrets=secrets) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, secrets=secrets) for item in value]
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = redact_value(item, secrets=secrets)
        return redacted
    return value


def _is_secret_key(value: str) -> bool:
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return bool(
        re.search(
            r"(?:^|[_-])(?:token|secret|password|authorization)(?:$|[_-])"
            r"|(?:^|[_-])api[_-]?key(?:$|[_-])",
            normalized,
        )
    )


class ArtifactStore:
    """Manage the fixed artifact layout below one application data directory."""

    def __init__(self, root: str | Path, *, secrets: tuple[str, ...] = ()) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._secrets = tuple(item for item in secrets if item)
        self._trace_lock = threading.Lock()

    def initialize(self, run_id: str) -> Path:
        run_dir = self._run_dir(run_id)
        try:
            run_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise ArtifactError(f"artifact directory already exists for {run_id}") from exc
        checks = run_dir / ARTIFACT_FILENAMES["checks"]
        checks.mkdir(mode=0o700)
        self._atomic_write(run_dir / ARTIFACT_FILENAMES["patch"], "")
        self._atomic_write(run_dir / ARTIFACT_FILENAMES["report"], "")
        self._atomic_write(run_dir / ARTIFACT_FILENAMES["result"], "{}\n")
        self._atomic_write(run_dir / ARTIFACT_FILENAMES["trace"], "")
        return run_dir

    def write_record(self, record: RunRecord) -> Path:
        payload = redact_value(record.model_dump(mode="json"), secrets=self._secrets)
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        path = self.path(record.run_id, "result")
        self._atomic_write(path, text)
        return path

    def write_patch(self, run_id: str, patch: str) -> Path:
        path = self.path(run_id, "patch")
        self._atomic_write(path, redact_text(patch, secrets=self._secrets))
        return path

    def write_report(self, run_id: str, report: str) -> Path:
        path = self.path(run_id, "report")
        self._atomic_write(path, redact_text(report, secrets=self._secrets))
        return path

    def write_check(self, run_id: str, name: str, text: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\.log", name):
            raise ArtifactError("invalid check artifact name")
        checks = self.path(run_id, "checks")
        target = checks / name
        self._assert_child(target, checks)
        self._atomic_write(target, redact_text(text, secrets=self._secrets))
        return target

    def append_trace(self, run_id: str, event: Mapping[str, Any]) -> Path:
        payload = dict(event)
        payload.setdefault("timestamp", utc_now())
        safe = redact_value(payload, secrets=self._secrets)
        line = json.dumps(
            safe,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._trace_lock:
            target = self.path(run_id, "trace")
            current = target.read_text(encoding="utf-8")
            self._atomic_write(target, current + line + "\n")
        return target

    def path(self, run_id: str, kind: ArtifactKind) -> Path:
        if kind not in ARTIFACT_FILENAMES:
            raise ArtifactError(f"unknown artifact kind: {kind}")
        run_dir = self._run_dir(run_id)
        target = run_dir / ARTIFACT_FILENAMES[kind]
        self._assert_child(target, run_dir)
        if not target.exists():
            raise ArtifactError(f"artifact does not exist: {kind}")
        if target.is_symlink():
            raise ArtifactError("artifact symlinks are not allowed")
        return target

    def _run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ArtifactError("invalid run id")
        target = self.root / run_id
        self._assert_child(target, self.root)
        return target

    @staticmethod
    def _assert_child(target: Path, parent: Path) -> None:
        try:
            target.resolve(strict=False).relative_to(parent.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ArtifactError("artifact path escapes its run directory") from exc

    @staticmethod
    def _atomic_write(target: Path, text: str) -> None:
        if target.exists() and target.is_symlink():
            raise ArtifactError("artifact symlinks are not allowed")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


__all__ = [
    "ArtifactError",
    "ArtifactStore",
    "redact_text",
    "redact_value",
]
