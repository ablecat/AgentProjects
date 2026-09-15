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
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

_PATCH_SECRET_PATTERNS = _SECRET_PATTERNS
_MIN_CONFIGURED_SECRET_SCAN_LENGTH = 8
_PATCH_ASSIGNMENT_PATTERN = re.compile(
    r"(?im)(?<![A-Za-z0-9_$-])"
    r"(?P<quote>[\"']?)(?P<name>[A-Za-z][A-Za-z0-9_-]{0,127})(?P=quote)"
    r"\s*(?P<separator>:(?!=)|=(?!=|>))\s*(?P<value>[^\r\n]*)"
)
_PATCH_SUBSCRIPT_ASSIGNMENT_PATTERN = re.compile(
    r"(?im)(?<![A-Za-z0-9_$-])(?:[A-Za-z_][A-Za-z0-9_.]*)"
    r"\[\s*(?P<quote>[\"'])(?P<name>[A-Za-z][A-Za-z0-9_-]{0,127})"
    r"(?P=quote)\s*\]\s*(?P<separator>=(?!=|>))\s*(?P<value>[^\r\n]*)"
)
_PATCH_POWERSHELL_ASSIGNMENT_PATTERN = re.compile(
    r"(?im)(?<![A-Za-z0-9_$-])\$(?:env:)?(?P<brace>\{)?\s*"
    r"(?P<name>[A-Za-z][A-Za-z0-9_-]{0,127})\s*(?(brace)\})"
    r"\s*(?P<separator>=(?!=|>))\s*(?P<value>[^\r\n]*)"
)
_PATCH_ASSIGNMENT_PATTERNS = (
    _PATCH_POWERSHELL_ASSIGNMENT_PATTERN,
    _PATCH_SUBSCRIPT_ASSIGNMENT_PATTERN,
    _PATCH_ASSIGNMENT_PATTERN,
)
_PATCH_SECRET_LITERAL_CALL_PATTERN = re.compile(
    r"(?im)(?:os\.putenv|os\.environ\.setdefault)\(\s*"
    r"(?P<key_quote>[\"'])(?P<name>[A-Za-z][A-Za-z0-9_-]{0,127})"
    r"(?P=key_quote)\s*,\s*(?P<value_quote>[\"'])"
    r"(?P<value>(?:\\.|[^\\\r\n])*?)(?P=value_quote)"
)
_SAFE_PATCH_VALUE_MARKERS = frozenset(
    {
        "",
        "***",
        "<api-key>",
        "<password>",
        "<redacted>",
        "<secret>",
        "<token>",
        "[redacted]",
        "changeme",
        "dummy",
        "example",
        "fake",
        "none",
        "null",
        "placeholder",
        "redacted",
        "replace-me",
        "replace_me",
        "test",
        "undefined",
        "your-api-key",
        "your-password",
        "your-secret",
        "your-token",
    }
)
_SAFE_PATCH_REFERENCE_PATTERNS = (
    re.compile(r"(?:os\.)?getenv\(\s*[\"'][A-Z_][A-Z0-9_]*[\"']\s*\)"),
    re.compile(r"System\.getenv\(\s*[\"'][A-Z_][A-Z0-9_]*[\"']\s*\)"),
    re.compile(r"os\.environ\[\s*[\"'][A-Z_][A-Z0-9_]*[\"']\s*\]"),
    re.compile(
        r"(?:config|env|process\.env|request|self|settings)\."
        r"[A-Za-z_][A-Za-z0-9_]*"
    ),
    re.compile(
        r"(?:config|request|settings)\[\s*[\"'][A-Za-z_][A-Za-z0-9_]*"
        r"[\"']\s*\]"
    ),
    re.compile(r"\$(?:env:)?[A-Za-z_][A-Za-z0-9_]*"),
    re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}"),
)
_SAFE_PATCH_IDENTIFIERS = frozenset(
    {"api_key", "credential", "key", "password", "secret", "token", "value"}
)
_SAFE_PATCH_TYPE = (
    r"(?:str|bytes|SecretStr|string)(?:\s*\|\s*(?:None|undefined|null))*"
)
_SAFE_PATCH_TYPE_PATTERN = re.compile(_SAFE_PATCH_TYPE)
_PATCH_TYPED_ASSIGNMENT_PATTERN = re.compile(
    rf"{_SAFE_PATCH_TYPE}\s*=(?!=|>)\s*(?P<value>.*)"
)
_PATCH_TYPED_OPERATOR_PATTERN = re.compile(
    rf"{_SAFE_PATCH_TYPE}\s*(?:==|!=|<=|>=|=>|:=).*"
)


class ArtifactError(RuntimeError):
    """Raised when a run artifact cannot be created or safely resolved."""


def redact_text(value: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Remove configured and high-confidence credential forms from text."""

    result = value
    for secret in sorted(
        (
            item
            for item in secrets
            if len(item) >= _MIN_CONFIGURED_SECRET_SCAN_LENGTH
        ),
        key=len,
        reverse=True,
    ):
        result = result.replace(secret, "[REDACTED]")
    result = _SECRET_PATTERNS[0].sub("Bearer [REDACTED]", result)
    result = _SECRET_PATTERNS[1].sub("[REDACTED]", result)
    result = _SECRET_PATTERNS[2].sub("[REDACTED]", result)
    result = _redact_secret_assignments(result)
    result = _redact_secret_literal_calls(result)
    return result


def patch_contains_credential(value: str, *, secrets: tuple[str, ...] = ()) -> bool:
    """Return whether an executable patch contains high-confidence credentials."""

    if any(_contains_configured_secret(value, secret) for secret in secrets if secret):
        return True
    if any(pattern.search(value) is not None for pattern in _PATCH_SECRET_PATTERNS):
        return True
    for pattern in _PATCH_ASSIGNMENT_PATTERNS:
        for match in pattern.finditer(value):
            if not _is_secret_key(match.group("name")):
                continue
            candidate = _patch_assignment_value(match)
            if candidate is None:
                continue
            if not _safe_patch_assignment_value(candidate):
                return True
    for match in _PATCH_SECRET_LITERAL_CALL_PATTERN.finditer(value):
        if _secret_literal_call_contains_credential(match):
            return True
    return False


def _contains_configured_secret(value: str, secret: str) -> bool:
    return len(secret) >= _MIN_CONFIGURED_SECRET_SCAN_LENGTH and secret in value


def _safe_patch_type_value(value: str) -> bool:
    candidate = value.strip().removesuffix(",").removesuffix(";").strip()
    return _SAFE_PATCH_TYPE_PATTERN.fullmatch(candidate) is not None


def _patch_assignment_value(match: re.Match[str]) -> str | None:
    candidate = match.group("value").strip()
    if match.group("separator") != ":":
        return candidate
    typed_assignment = _PATCH_TYPED_ASSIGNMENT_PATTERN.fullmatch(candidate)
    if typed_assignment is not None:
        return typed_assignment.group("value").strip()
    if _safe_patch_type_value(candidate):
        return None
    if _PATCH_TYPED_OPERATOR_PATTERN.fullmatch(candidate) is not None:
        return None
    return candidate


def _secret_literal_call_contains_credential(match: re.Match[str]) -> bool:
    if not _is_secret_key(match.group("name")):
        return False
    quote = match.group("value_quote")
    return not _safe_patch_assignment_value(quote + match.group("value") + quote)


def _safe_patch_assignment_value(value: str) -> bool:
    candidate = value.strip().removesuffix(",").removesuffix(";").strip()
    quoted = (
        len(candidate) >= 2
        and candidate[0] in {"'", '"'}
        and candidate[-1] == candidate[0]
    )
    normalized = candidate[1:-1] if quoted else candidate
    folded = normalized.casefold()
    if folded in _SAFE_PATCH_VALUE_MARKERS:
        return True
    if quoted:
        return False
    if candidate == "...":
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", candidate):
        return True
    if any(pattern.fullmatch(candidate) for pattern in _SAFE_PATCH_REFERENCE_PATTERNS):
        return True
    return candidate in _SAFE_PATCH_IDENTIFIERS


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
    normalized = value.replace("-", "_")
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", normalized)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized).casefold()
    return bool(
        re.search(
            r"(?:^|[_-])(?:token|secret|password|passwd|authorization)"
            r"(?:$|[_-])|(?:^|[_-])(?:api|private)[_-]?key(?:$|[_-])",
            normalized,
        )
    )


def _redact_secret_assignments(value: str) -> str:
    chunks: list[str] = []
    cursor = 0
    matches = sorted(
        (
            match
            for pattern in _PATCH_ASSIGNMENT_PATTERNS
            for match in pattern.finditer(value)
            if _is_secret_key(match.group("name"))
            and _patch_assignment_value(match) is not None
        ),
        key=lambda match: (match.start(), -match.end()),
    )
    for match in matches:
        if match.start() < cursor:
            continue
        chunks.append(value[cursor : match.start("value")])
        chunks.append("[REDACTED]")
        cursor = match.end("value")
    if not chunks:
        return value
    chunks.append(value[cursor:])
    return "".join(chunks)


def _redact_secret_literal_calls(value: str) -> str:
    chunks: list[str] = []
    cursor = 0
    for match in _PATCH_SECRET_LITERAL_CALL_PATTERN.finditer(value):
        if not _secret_literal_call_contains_credential(match):
            continue
        chunks.append(value[cursor : match.start("value_quote")])
        chunks.append("[REDACTED]")
        cursor = match.end()
    if not chunks:
        return value
    chunks.append(value[cursor:])
    return "".join(chunks)


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
        if patch_contains_credential(patch, secrets=self._secrets):
            raise ArtifactError("candidate patch contains credential-like content")
        self._atomic_write(path, patch)
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
    "patch_contains_credential",
    "redact_text",
    "redact_value",
]
