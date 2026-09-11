from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest

from repo_agent.checks import detect_check_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_validator() -> ModuleType:
    path = PROJECT_ROOT / "benchmarks" / "validate.py"
    spec = importlib.util.spec_from_file_location("repo_agent_benchmark_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_day3_python_java_benchmark_contracts_and_content_locks() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(PROJECT_ROOT / "benchmarks" / "validate.py"),
            "--structure-only",
        ),
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        timeout=30,
        shell=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert "[ok] benchmark structure, contracts, and locks" in completed.stdout


def test_task_digest_is_stable_across_line_endings_and_generated_caches(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    task_dir = tmp_path / "task"
    source = task_dir / "baseline" / "tasklib" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"value = 1\n")
    expected = validator.task_digest(task_dir)

    source.write_bytes(b"value = 1\r\n")
    cache = source.parent / "__pycache__"
    cache.mkdir()
    (cache / "example.cpython-311.pyc").write_bytes(b"generated")
    (task_dir / ".pytest_cache").mkdir()
    (task_dir / ".pytest_cache" / "README.md").write_text(
        "generated cache", encoding="utf-8"
    )

    assert validator.task_digest(task_dir) == expected

    source.write_bytes(b"value = 2\n")
    assert validator.task_digest(task_dir) != expected


def test_task_digest_rejects_generated_looking_symlinks(tmp_path: Path) -> None:
    validator = _load_validator()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    target = tmp_path / "generated.pyc"
    target.write_bytes(b"generated")
    link = task_dir / "ignored.pyc"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(validator.ValidationError, match="symlinks are not allowed"):
        validator.task_digest(task_dir)


def test_maven_bootstrap_and_checks_have_separate_network_phases(
    monkeypatch, tmp_path: Path
) -> None:
    validator = _load_validator()
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cache"
    worktree.mkdir()
    cache.mkdir()
    commands: list[tuple[str, ...]] = []
    container_id = "a" * 64

    def fake_capture(argv, timeout):
        del timeout
        command = tuple(argv)
        commands.append(command)
        if command[1:3] == ("container", "create"):
            return subprocess.CompletedProcess(command, 0, container_id + "\n")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(validator, "_docker_capture", fake_capture)

    validator.bootstrap_maven_dependencies(worktree, cache, 180)
    bootstrap_creates = [
        command for command in commands if command[1:3] == ("container", "create")
    ]
    assert len(bootstrap_creates) == 4
    bootstrap_commands = [
        command[command.index(validator.MAVEN_IMAGE) + 1 :]
        for command in bootstrap_creates
    ]
    assert all(
        command[command.index("--network") + 1] == "bridge"
        for command in bootstrap_creates
    )
    assert (
        "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:go-offline"
        in bootstrap_commands[0]
    )
    assert "-Prepo-agent-bootstrap" not in bootstrap_commands[0]
    assert "-DskipTests" in bootstrap_commands[0]
    assert {
        argument.removeprefix("-Dartifact=")
        for command in bootstrap_commands[1:]
        for argument in command
        if argument.startswith("-Dartifact=")
    } == {
        "org.apache.maven.surefire:surefire-junit-platform:3.5.2",
        "org.junit.platform:junit-platform-launcher:1.9.3",
        "org.junit.platform:junit-platform-launcher:1.11.4",
    }
    for command in bootstrap_commands:
        assert "test" not in command
        assert not any(argument.startswith("-Dtest=") for argument in command)

    commands.clear()
    validator.run_maven_tests(worktree, cache, ["ExampleTest"], 30)
    check_create = commands[0]
    check_maven = check_create[check_create.index(validator.MAVEN_IMAGE) + 1 :]
    assert check_create[check_create.index("--network") + 1] == "none"
    assert "--offline" in check_maven
    assert "-Dtest=ExampleTest" in check_maven
    assert check_maven[-1] == "test"


def test_maven_create_timeout_attempts_exact_name_cleanup(
    monkeypatch, tmp_path: Path
) -> None:
    validator = _load_validator()
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cache"
    worktree.mkdir()
    cache.mkdir()
    commands: list[tuple[str, ...]] = []
    container_id = "b" * 64

    def fake_capture(argv, timeout):
        command = tuple(argv)
        commands.append(command)
        if command[1:3] == ("container", "create"):
            raise subprocess.TimeoutExpired(command, timeout)
        if command[1:3] == ("container", "inspect"):
            return subprocess.CompletedProcess(command, 0, f"{container_id}|true\n")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(validator, "_docker_capture", fake_capture)

    with pytest.raises(subprocess.TimeoutExpired):
        validator.bootstrap_maven_dependencies(worktree, cache, 180)

    assert len(commands) == 6
    for offset in (0, 3):
        generated_name = commands[offset][commands[offset].index("--name") + 1]
        assert generated_name.startswith("repo-agent-benchmark-")
        assert len(generated_name.removeprefix("repo-agent-benchmark-")) == 32
        assert commands[offset + 1][1:3] == ("container", "inspect")
        assert commands[offset + 1][-1] == generated_name
        assert commands[offset + 2] == (
            "docker",
            "container",
            "rm",
            "--force",
            container_id,
        )


def test_maven_bootstrap_retries_once_with_the_same_cache(
    monkeypatch, tmp_path: Path
) -> None:
    validator = _load_validator()
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cache"
    worktree.mkdir()
    cache.mkdir()
    commands: list[tuple[str, ...]] = []
    container_ids = iter(character * 64 for character in "efabcd")
    starts = 0

    def fake_capture(argv, timeout):
        nonlocal starts
        del timeout
        command = tuple(argv)
        commands.append(command)
        if command[1:3] == ("container", "create"):
            return subprocess.CompletedProcess(command, 0, next(container_ids) + "\n")
        if command[1:3] == ("container", "start"):
            starts += 1
            return subprocess.CompletedProcess(
                command,
                1 if starts == 1 else 0,
                (
                    "Could not transfer artifact example:fixture:jar:1 from/to "
                    "central (https://repo.maven.apache.org/maven2): Premature end "
                    "of Content-Length delimited message body (expected: 246,918; "
                    "received: 229,376)"
                    if starts == 1
                    else "downloaded"
                ),
            )
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(validator, "_docker_capture", fake_capture)

    result = validator.bootstrap_maven_dependencies(worktree, cache, 180)

    assert result.returncode == 0
    assert "Premature end of Content-Length" in result.stdout
    assert "downloaded" in result.stdout
    creates = [
        command for command in commands if command[1:3] == ("container", "create")
    ]
    assert len(creates) == 5
    for command in creates:
        assert command[command.index("--network") + 1] == "bridge"
        mounts = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--mount"
        ]
        assert any(f"source={cache.resolve()}" in mount for mount in mounts)


@pytest.mark.parametrize(
    "failure",
    [
        (
            "Could not transfer artifact example:fixture:jar:1 from/to central "
            "(https://repo.maven.apache.org/maven2): Remote host terminated the "
            "handshake: SSL peer shut down incorrectly"
        ),
        (
            "Could not transfer artifact example:fixture:jar:1 from/to central: "
            "java.net.SocketTimeoutException: Read timed out"
        ),
        (
            "Could not transfer metadata example:fixture/maven-metadata.xml "
            "from/to central: status code: 503, reason phrase: Service Unavailable"
        ),
    ],
)
def test_maven_bootstrap_retries_transient_transfer_failure(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    validator = _load_validator()
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cache"
    worktree.mkdir()
    cache.mkdir()
    calls: list[tuple[Path, Path, tuple[str, ...], int, str]] = []

    def fake_run(worktree_arg, cache_arg, maven_args, timeout, *, network):
        calls.append((worktree_arg, cache_arg, tuple(maven_args), timeout, network))
        call = tuple(maven_args)
        if len(calls) == 1:
            return subprocess.CompletedProcess(call, 1, failure)
        return subprocess.CompletedProcess(call, 0, "downloaded")

    monkeypatch.setattr(validator, "_run_maven_container", fake_run)

    result = validator.bootstrap_maven_dependencies(worktree, cache, 180)

    assert result.returncode == 0
    assert failure in result.stdout
    assert "downloaded" in result.stdout
    assert len(calls) == 5
    assert calls[0] == calls[1]
    assert all(call[1] == cache and call[4] == "bridge" for call in calls)


def test_maven_bootstrap_stops_after_one_transient_retry(
    monkeypatch, tmp_path: Path
) -> None:
    validator = _load_validator()
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cache"
    worktree.mkdir()
    cache.mkdir()
    calls = 0

    def fake_run(worktree_arg, cache_arg, maven_args, timeout, *, network):
        nonlocal calls
        del worktree_arg, cache_arg, timeout, network
        calls += 1
        output = (
            "Could not transfer artifact example:fixture:jar:1 from/to central: "
            f"Connection reset (attempt {calls})"
        )
        return subprocess.CompletedProcess(tuple(maven_args), 1, output)

    monkeypatch.setattr(validator, "_run_maven_container", fake_run)

    result = validator.bootstrap_maven_dependencies(worktree, cache, 180)

    assert result.returncode == 1
    assert calls == 2
    assert "attempt 1" in result.stdout
    assert "attempt 2" in result.stdout


@pytest.mark.parametrize(
    "failure",
    [
        "Could not resolve artifact example:missing:jar:1",
        (
            "Could not transfer artifact example:fixture:jar:1 from/to central: "
            "Premature end of Content-Length delimited message body "
            "(expected: 10; received: 10)"
        ),
        "Cannot access central in offline mode",
        "[ERROR] COMPILATION ERROR: cannot find symbol",
        "[ERROR] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0",
        (
            "Could not transfer artifact example:fixture:jar:1 from/to central: "
            "PKIX path building failed: unable to find valid certification path"
        ),
        "Remote host terminated the handshake: SSL peer shut down incorrectly",
    ],
)
def test_maven_bootstrap_does_not_retry_deterministic_failures(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    validator = _load_validator()
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cache"
    worktree.mkdir()
    cache.mkdir()
    commands: list[tuple[str, ...]] = []
    container_id = "f" * 64

    def fake_capture(argv, timeout):
        del timeout
        command = tuple(argv)
        commands.append(command)
        if command[1:3] == ("container", "create"):
            return subprocess.CompletedProcess(command, 0, container_id + "\n")
        if command[1:3] == ("container", "start"):
            return subprocess.CompletedProcess(command, 1, failure)
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(validator, "_docker_capture", fake_capture)

    result = validator.bootstrap_maven_dependencies(worktree, cache, 180)

    assert result.returncode == 1
    creates = [
        command for command in commands if command[1:3] == ("container", "create")
    ]
    assert len(creates) == 1


@pytest.mark.parametrize(
    "injection",
    [
        (
            "</project>",
            "<repositories><repository><id>evil</id>"
            "<url>https://example.invalid/maven</url></repository></repositories>"
            "</project>",
        ),
        (
            "<plugins>",
            "<extensions><extension><groupId>evil</groupId><artifactId>loader</artifactId>"
            "<version>1</version></extension></extensions><plugins>",
        ),
        (
            "</plugins>",
            "<plugin><groupId>evil</groupId><artifactId>runner</artifactId>"
            "<version>1</version></plugin></plugins>",
        ),
    ],
)
def test_java_layout_rejects_network_or_plugin_injection(
    tmp_path: Path, injection: tuple[str, str]
) -> None:
    validator = _load_validator()
    source = (
        PROJECT_ROOT
        / "benchmarks"
        / "tasks"
        / "java"
        / "bugfix"
        / "backoff-attempt"
    )
    task = tmp_path / "backoff-attempt"
    shutil.copytree(source, task)
    pom = task / "baseline" / "pom.xml"
    before, after = injection
    pom.write_text(
        pom.read_text(encoding="utf-8").replace(before, after, 1),
        encoding="utf-8",
    )

    with pytest.raises(validator.ValidationError, match="unexpected Maven"):
        validator._validate_java_layout(task, "java-bugfix-test")


def test_java_layout_rejects_project_maven_configuration(tmp_path: Path) -> None:
    validator = _load_validator()
    source = (
        PROJECT_ROOT
        / "benchmarks"
        / "tasks"
        / "java"
        / "bugfix"
        / "backoff-attempt"
    )
    task = tmp_path / "backoff-attempt"
    shutil.copytree(source, task)
    maven_config = task / "baseline" / ".mvn"
    maven_config.mkdir()
    (maven_config / "extensions.xml").write_text("<extensions />", encoding="utf-8")

    with pytest.raises(validator.ValidationError, match=r"\.mvn configuration"):
        validator._validate_java_layout(task, "java-bugfix-test")


@pytest.mark.parametrize(
    "slug",
    [
        "backoff-attempt",
        "page-zero",
        "shared-default-labels",
        "malformed-retry-header",
        "dotfile-extension",
        "case-insensitive-labels",
    ],
)
def test_java_benchmark_poms_match_the_production_check_profile(
    slug: str, tmp_path: Path
) -> None:
    validator = _load_validator()
    task = PROJECT_ROOT / "benchmarks" / "tasks" / "java" / "bugfix" / slug
    baseline = task / "baseline"
    worktree = tmp_path / slug
    shutil.copytree(baseline, worktree)

    for stage, patch_name in (
        ("baseline", None),
        ("setup", "setup.patch"),
        ("gold", "gold.patch"),
    ):
        if patch_name is not None:
            validator.apply_patch(worktree, task / patch_name, f"{slug}-{stage}")
        profile = detect_check_profile(worktree)
        assert profile.id == "maven-test"
        assert profile.java_release == 17
        assert len(profile.bootstrap_argv) == 4
        assert "--offline" in profile.check_argv


def test_all_task_metadata_execute_the_committed_json_schema() -> None:
    validator = _load_validator()
    schema = validator.load_json(validator.TASK_SCHEMA_PATH)
    manifest = validator.load_json(validator.MANIFEST_PATH)
    entries = validator.validate_manifest(manifest)

    validated = []
    for entry in entries:
        task_dir = validator.safe_path(entry["path"])
        metadata = validator.validate_metadata(task_dir, entry, schema)
        validated.append(metadata["id"])

    assert validated == [entry["id"] for entry in entries]
    assert len(validated) == 12


@pytest.mark.parametrize(
    ("task_id", "invalid_signature"),
    [
        ("py-bugfix-001", "missing_test_prefix"),
        ("java-bugfix-001", "not-a-java-method"),
    ],
)
def test_json_schema_enforces_language_specific_failure_signatures(
    task_id: str, invalid_signature: str
) -> None:
    validator = _load_validator()
    schema = validator.load_json(validator.TASK_SCHEMA_PATH)
    manifest = validator.load_json(validator.MANIFEST_PATH)
    entry = next(item for item in manifest["tasks"] if item["id"] == task_id)
    metadata = validator.load_json(validator.safe_path(entry["path"]) / "metadata.json")
    invalid = copy.deepcopy(metadata)
    invalid["test_contract"]["failure_signatures"] = [invalid_signature]

    with pytest.raises(validator.ValidationError, match="schema pattern"):
        validator.validate_json_schema(invalid, schema)


@pytest.mark.parametrize(
    ("remove_code", "remove_output", "raises"),
    [
        (1, "permission denied", True),
        (1, "Error response from daemon: No such container: fixture", False),
        (2, "Error response from daemon: No such container: fixture", True),
    ],
)
def test_maven_container_cleanup_only_accepts_explicit_absence(
    monkeypatch,
    tmp_path: Path,
    remove_code: int,
    remove_output: str,
    raises: bool,
) -> None:
    validator = _load_validator()
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cache"
    worktree.mkdir()
    cache.mkdir()
    container_id = "9" * 64

    def fake_capture(argv, timeout):
        del timeout
        command = tuple(argv)
        if command[1:3] == ("container", "create"):
            return subprocess.CompletedProcess(command, 0, container_id + "\n")
        if command[1:3] == ("container", "start"):
            return subprocess.CompletedProcess(command, 0, "tests passed")
        if command[1:3] == ("container", "rm"):
            return subprocess.CompletedProcess(command, remove_code, remove_output)
        raise AssertionError(command)

    monkeypatch.setattr(validator, "_docker_capture", fake_capture)

    if raises:
        with pytest.raises(validator.ValidationError, match="failed to remove"):
            validator.run_maven_tests(worktree, cache, ["ExampleTest"], 30)
    else:
        result = validator.run_maven_tests(worktree, cache, ["ExampleTest"], 30)
        assert result.returncode == 0


def test_unresolved_container_cleanup_refuses_wrong_owner(monkeypatch) -> None:
    validator = _load_validator()
    commands: list[tuple[str, ...]] = []
    container_id = "c" * 64
    generated_name = "repo-agent-benchmark-" + "d" * 32

    def fake_capture(argv, timeout):
        del timeout
        command = tuple(argv)
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, f"{container_id}|false\n")

    monkeypatch.setattr(validator, "_docker_capture", fake_capture)
    validator._remove_generated_container(generated_name, unresolved_name=True)

    assert len(commands) == 1
    assert commands[0][1:3] == ("container", "inspect")


def test_disposable_tree_is_recursively_writable_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    disposable = tmp_path / "disposable"
    nested = disposable / "nested"
    nested.mkdir(parents=True)
    source = nested / "Example.java"
    source.write_text("final class Example {}\n", encoding="utf-8")
    source.chmod(0o444)

    validator.prepare_container_writable_tree(disposable)

    assert disposable.stat().st_mode & 0o222
    assert nested.stat().st_mode & 0o222
    assert source.stat().st_mode & 0o222

    link = disposable / "linked.java"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(validator.ValidationError, match="symlink"):
        validator.prepare_container_writable_tree(disposable)
