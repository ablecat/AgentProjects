from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

import pytest

import repo_agent.checks as checks_module
from repo_agent.checks import (
    MAX_CHECK_OUTPUT_BYTES,
    CheckPolicyError,
    CheckRunner,
    detect_check_profile,
)
from repo_agent.patches import PatchValidationError
from repo_agent.sandbox import CommandOutcome


PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "check-fixture"
version = "0.1.0"
dependencies = []
"""

MAVEN_POM = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>example</groupId>
  <artifactId>check-fixture</artifactId>
  <version>1.0.0</version>
  <properties>
    <maven.compiler.release>{release}</maven.compiler.release>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.11.4</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.5.2</version>
      </plugin>
    </plugins>
  </build>
</project>
"""

MAVEN_PLUGIN_POM = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>example</groupId>
  <artifactId>check-fixture</artifactId>
  <version>1.0.0</version>
  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.11.4</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-compiler-plugin</artifactId>
        <version>3.13.0</version>
        <configuration>
          <release>{release}</release>
        </configuration>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.5.2</version>
      </plugin>
    </plugins>
  </build>
</project>
"""

CONTAINER_IDS = tuple(f"{number:064x}" for number in range(1, 10))


class FakeDockerRunner:
    def __init__(
        self,
        *,
        starts: list[CommandOutcome] | None = None,
        creates: list[CommandOutcome] | None = None,
        inspects: list[CommandOutcome] | None = None,
        inspect_owned: bool = False,
        on_start: Callable[[tuple[str, ...]], None] | None = None,
        start_exception: BaseException | None = None,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []
        self.output_limits: list[int] = []
        self.starts = list(starts or [])
        self.creates = list(creates or [])
        self.inspects = list(inspects or [])
        self.inspect_owned = inspect_owned
        self.on_start = on_start
        self.start_exception = start_exception
        self._create_count = 0
        self._create_by_id: dict[str, tuple[str, ...]] = {}
        self._create_by_name: dict[str, tuple[str, ...]] = {}

    def __call__(
        self,
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        command = tuple(argv)
        self.commands.append(command)
        self.timeouts.append(timeout_seconds)
        self.output_limits.append(max_output_bytes)
        operation = command[2]
        if operation == "create":
            self._create_count += 1
            container_id = CONTAINER_IDS[self._create_count - 1]
            name = command[command.index("--name") + 1]
            self._create_by_id[container_id] = command
            self._create_by_name[name] = command
            if self.creates:
                return self.creates.pop(0)
            return CommandOutcome(0, f"{container_id}\n")
        if operation == "start":
            container_id = command[-1]
            if self.on_start is not None:
                self.on_start(self._create_by_id[container_id])
            if self.start_exception is not None:
                raise self.start_exception
            if self.starts:
                return self.starts.pop(0)
            return CommandOutcome(0, "phase output\n")
        if operation == "inspect":
            name = command[-1]
            if self.inspects:
                return self.inspects.pop(0)
            if not self.inspect_owned:
                return CommandOutcome(1, "Error: No such container\n")
            create = self._create_by_name[name]
            labels = [
                create[index + 1]
                for index, value in enumerate(create)
                if value == "--label"
            ]
            run_id = labels[0].split("=", 1)[1]
            sequence = labels[1].split("=", 1)[1]
            container_id = CONTAINER_IDS[self._create_count - 1]
            return CommandOutcome(
                0,
                f"repo-agent-check-owned {container_id} {run_id} {sequence}\n",
            )
        if operation == "rm":
            return CommandOutcome(0, f"{command[-1]}\n")
        raise AssertionError(f"unexpected Docker command: {command}")


def _option_values(command: tuple[str, ...], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def _mount_sources(command: tuple[str, ...]) -> dict[str, str]:
    mounts: dict[str, str] = {}
    for mount in _option_values(command, "--mount"):
        fields = dict(item.split("=", 1) for item in mount.split(",") if "=" in item)
        mounts[fields["target"]] = fields["source"]
    return mounts


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _commit_repository(repo: Path) -> Path:
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Check Test")
    _git(repo, "config", "user.email", "check-test@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo.resolve()


def _python_repository(tmp_path: Path, *, requirements: str | None = None) -> Path:
    repo = tmp_path / "python-source"
    (repo / "tests").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    if requirements is not None:
        (repo / "requirements.txt").write_text(requirements, encoding="utf-8")
    return _commit_repository(repo)


def _maven_repository(tmp_path: Path, *, release: str = "17") -> Path:
    repo = tmp_path / "maven-source"
    repo.mkdir()
    (repo / "pom.xml").write_text(
        MAVEN_POM.format(release=release), encoding="utf-8"
    )
    return _commit_repository(repo)


def _maven_plugin_repository(tmp_path: Path, name: str, release: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / "pom.xml").write_text(
        MAVEN_PLUGIN_POM.format(release=release), encoding="utf-8"
    )
    return _commit_repository(repo)


def _repository_with_pom(tmp_path: Path, name: str, pom: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / "pom.xml").write_text(pom, encoding="utf-8")
    return _commit_repository(repo)


def test_clean_crlf_checkout_uses_effective_autocrlf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_home = tmp_path / "git-home"
    git_home.mkdir()
    (git_home / ".gitconfig").write_text(
        "[core]\n\tautocrlf = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(git_home))
    repo = tmp_path / "crlf-maven"
    repo.mkdir()
    pom = MAVEN_POM.format(release="17").replace("\n", "\r\n")
    (repo / "pom.xml").write_bytes(pom.encode("utf-8"))
    _commit_repository(repo)
    docker = FakeDockerRunner()

    result = CheckRunner(
        repo,
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "passed"
    assert result.cleanup_ok
    assert any(command[2] == "create" for command in docker.commands)


def test_autocrlf_lookup_failure_is_a_structured_setup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _python_repository(tmp_path)

    def fail_lookup(*_args, **_kwargs):
        raise RuntimeError("could not read line-ending configuration")

    monkeypatch.setattr(checks_module, "effective_core_autocrlf", fail_lookup)
    docker = FakeDockerRunner()

    result = CheckRunner(
        repo,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "setup_error"
    assert "could not read line-ending configuration" in (result.error or "")
    assert docker.commands == []


def test_detects_python_pytest_profile_without_bootstrap(tmp_path: Path) -> None:
    profile = detect_check_profile(_python_repository(tmp_path))

    assert profile.id == "python-pytest"
    assert profile.language == "python"
    assert profile.image == "repo-agent-python:0.1"
    assert profile.manifest == "pyproject.toml"
    assert profile.bootstrap_argv == ()
    assert profile.check_argv == (
        "python",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    )


def test_python_requirements_enable_only_preregistered_bootstrap_argv(
    tmp_path: Path,
) -> None:
    profile = detect_check_profile(
        _python_repository(tmp_path, requirements="requests==2.32.5\n")
    )

    assert profile.bootstrap_required
    assert profile.bootstrap_argv == (
        (
            "python",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-compile",
            "--target",
            "/dependencies/python",
            "--requirement",
            "requirements.txt",
        ),
    )
    assert profile.check_argv[0] == "python"


def test_detects_single_module_maven_profile_with_offline_verify(
    tmp_path: Path,
) -> None:
    profile = detect_check_profile(_maven_repository(tmp_path, release="21"))

    assert profile.id == "maven-test"
    assert profile.language == "java"
    assert profile.image == "repo-agent-maven:0.1"
    assert profile.java_release == 21
    assert profile.bootstrap_argv == (
        (
            "mvn",
            "--batch-mode",
            "--no-transfer-progress",
            "-Dstyle.color=never",
            "-Dmaven.repo.local=/dependencies/m2",
            "-DskipTests",
            "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:go-offline",
        ),
        (
            "mvn",
            "--batch-mode",
            "--no-transfer-progress",
            "-Dstyle.color=never",
            "-Dmaven.repo.local=/dependencies/m2",
            "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
            "-Dartifact=org.apache.maven.surefire:surefire-junit-platform:3.5.2",
            "-Dtransitive=true",
        ),
        (
            "mvn",
            "--batch-mode",
            "--no-transfer-progress",
            "-Dstyle.color=never",
            "-Dmaven.repo.local=/dependencies/m2",
            "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
            "-Dartifact=org.junit.platform:junit-platform-launcher:1.9.3",
            "-Dtransitive=true",
        ),
        (
            "mvn",
            "--batch-mode",
            "--no-transfer-progress",
            "-Dstyle.color=never",
            "-Dmaven.repo.local=/dependencies/m2",
            "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
            "-Dartifact=org.junit.platform:junit-platform-launcher:1.11.4",
            "-Dtransitive=true",
        ),
    )
    assert "--offline" in profile.check_argv
    assert profile.check_argv[-2:] == (
        "process-test-classes",
        "org.apache.maven.plugins:maven-surefire-plugin:3.5.2:test",
    )


def test_maven_verify_forces_tests_to_run_and_fail_closed(tmp_path: Path) -> None:
    profile = detect_check_profile(_maven_repository(tmp_path, release="21"))

    required_overrides = {
        "-DfailIfNoTests=true",
        "-DskipTests=false",
        "-Dmaven.test.skip=false",
        "-Dmaven.test.failure.ignore=false",
    }
    assert required_overrides <= set(profile.check_argv)
    assert "-DskipTests" not in profile.check_argv
    assert "--offline" in profile.check_argv
    assert profile.check_argv[-2:] == (
        "process-test-classes",
        "org.apache.maven.plugins:maven-surefire-plugin:3.5.2:test",
    )


@pytest.mark.parametrize(
    ("element", "message"),
    [
        (
            "<parent><groupId>evil.example</groupId><artifactId>parent</artifactId>"
            "<version>1.0</version></parent>",
            "parent POMs",
        ),
        (
            "<repositories><repository><id>evil</id>"
            "<url>https://evil.invalid/maven</url></repository></repositories>",
            "repositories",
        ),
        (
            "<pluginRepositories><pluginRepository><id>evil</id>"
            "<url>https://evil.invalid/plugins</url></pluginRepository>"
            "</pluginRepositories>",
            "pluginRepositories",
        ),
        ("<profiles><profile><id>hidden</id></profile></profiles>", "profiles"),
        ("<modules />", "multi-module Maven"),
        (
            '<repositories xmlns=""><repository><id>hidden</id></repository>'
            "</repositories>",
            "unsupported XML namespace",
        ),
    ],
)
def test_maven_external_model_sources_are_denied_before_docker(
    tmp_path: Path, element: str, message: str
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "  <groupId>example</groupId>",
        f"  {element}\n  <groupId>example</groupId>",
        1,
    )
    docker = FakeDockerRunner()

    result = CheckRunner(
        _repository_with_pom(tmp_path, "maven-external-model", pom),
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert message in (result.error or "")
    assert docker.commands == []


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("extension", "build extensions"),
        ("dependency", "only depend on JUnit Jupiter"),
        ("plugin", "only use the fixed Surefire"),
        ("plugin-management", "pluginManagement"),
    ],
)
def test_maven_extra_build_inputs_are_denied_before_docker(
    tmp_path: Path, kind: str, message: str
) -> None:
    pom = MAVEN_POM.format(release="17")
    if kind == "extension":
        pom = pom.replace(
            "  <build>\n",
            "  <build>\n"
            "    <extensions><extension><groupId>evil.example</groupId>"
            "<artifactId>build-extension</artifactId><version>1.0</version>"
            "</extension></extensions>\n",
            1,
        )
    elif kind == "dependency":
        pom = pom.replace(
            "  </dependencies>\n",
            "    <dependency><groupId>evil.example</groupId>"
            "<artifactId>extra</artifactId><version>1.0</version>"
            "</dependency>\n  </dependencies>\n",
            1,
        )
    elif kind == "plugin":
        pom = pom.replace(
            "    </plugins>\n",
            "      <plugin><groupId>evil.example</groupId>"
            "<artifactId>exec-plugin</artifactId><version>1.0</version>"
            "</plugin>\n    </plugins>\n",
            1,
        )
    else:
        pom = pom.replace(
            "  <build>\n",
            "  <build>\n    <pluginManagement><plugins /></pluginManagement>\n",
            1,
        )
    docker = FakeDockerRunner()

    result = CheckRunner(
        _repository_with_pom(tmp_path, f"maven-extra-{kind}", pom),
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert message in (result.error or "")
    assert docker.commands == []


@pytest.mark.parametrize(
    "configuration",
    [
        "<skipTests>true</skipTests>",
        "<excludes><exclude>**/*Test.java</exclude></excludes>",
        "<includes><include>**/NeverRuns.java</include></includes>",
        "<testFailureIgnore>true</testFailureIgnore>",
        "<useModulePath>true</useModulePath>",
    ],
)
def test_maven_surefire_test_selection_is_denied_before_docker(
    tmp_path: Path, configuration: str
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "        <version>3.5.2</version>",
        "        <version>3.5.2</version>\n"
        f"        <configuration>{configuration}</configuration>",
        1,
    )
    docker = FakeDockerRunner()

    result = CheckRunner(
        _repository_with_pom(tmp_path, "maven-surefire-bypass", pom),
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert "Surefire" in (result.error or "")
    assert docker.commands == []


@pytest.mark.parametrize(
    "property_xml",
    [
        "<skipTests>true</skipTests>",
        "<test>NeverRuns</test>",
        "<maven.test.failure.ignore>true</maven.test.failure.ignore>",
    ],
)
def test_maven_test_control_properties_are_denied_before_docker(
    tmp_path: Path, property_xml: str
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "  </properties>", f"    {property_xml}\n  </properties>", 1
    )
    docker = FakeDockerRunner()

    result = CheckRunner(
        _repository_with_pom(tmp_path, "maven-test-property", pom),
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert "property" in (result.error or "")
    assert docker.commands == []


@pytest.mark.parametrize("property_name", ["maven.compiler.source", "java.version"])
def test_maven_non_release_java_properties_are_denied_before_docker(
    tmp_path: Path, property_name: str
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "<maven.compiler.release>17</maven.compiler.release>",
        f"<{property_name}>17</{property_name}>",
        1,
    )
    docker = FakeDockerRunner()

    result = CheckRunner(
        _repository_with_pom(
            tmp_path, f"maven-{property_name.replace('.', '-')}", pom
        ),
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert "property" in (result.error or "")
    assert docker.commands == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model-version", "modelVersion"),
        ("packaging", "packaging"),
        ("compiler-source", "compiler configuration"),
        ("compiler-version", "compiler plugin"),
    ],
)
def test_maven_project_and_compiler_contract_is_fixed_before_docker(
    tmp_path: Path, mutation: str, message: str
) -> None:
    pom = MAVEN_PLUGIN_POM.format(release="17")
    if mutation == "model-version":
        pom = pom.replace(
            "<modelVersion>4.0.0</modelVersion>",
            "<modelVersion>3.0.0</modelVersion>",
            1,
        )
    elif mutation == "packaging":
        pom = pom.replace(
            "  <version>1.0.0</version>",
            "  <version>1.0.0</version>\n  <packaging>maven-plugin</packaging>",
            1,
        )
    elif mutation == "compiler-source":
        pom = pom.replace("<release>17</release>", "<source>17</source>", 1)
    else:
        pom = pom.replace(
            "        <version>3.13.0</version>",
            "        <version>3.12.1</version>",
            1,
        )
    docker = FakeDockerRunner()

    result = CheckRunner(
        _repository_with_pom(tmp_path, f"maven-fixed-{mutation}", pom),
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert message in (result.error or "")
    assert docker.commands == []


def test_maven_utf8_and_safe_surefire_configuration_are_allowed(
    tmp_path: Path,
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "  <version>1.0.0</version>",
        "  <version>1.0.0</version>\n  <packaging>jar</packaging>",
        1,
    ).replace(
        "  </properties>",
        "    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>\n"
        "  </properties>",
        1,
    ).replace(
        "        <version>3.5.2</version>",
        "        <version>3.5.2</version>\n"
        "        <configuration><useModulePath>false</useModulePath></configuration>",
        1,
    )

    profile = detect_check_profile(
        _repository_with_pom(tmp_path, "maven-safe-surefire", pom)
    )

    assert profile.id == "maven-test"
    assert profile.java_release == 17


def test_maven_surefire_version_is_required(tmp_path: Path) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "        <version>3.5.2</version>\n", ""
    )
    repo = _repository_with_pom(tmp_path, "missing-surefire-version", pom)

    with pytest.raises(
        CheckPolicyError,
        match="maven-surefire-plugin must declare one explicit version",
    ):
        detect_check_profile(repo)


def test_maven_surefire_property_version_is_resolved_for_bootstrap(
    tmp_path: Path,
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "    <maven.compiler.release>17</maven.compiler.release>\n",
        "    <maven.compiler.release>17</maven.compiler.release>\n"
        "    <surefire.version>3.5.2</surefire.version>\n",
    ).replace(
        "        <version>3.5.2</version>",
        "        <version>${surefire.version}</version>",
    )
    repo = _repository_with_pom(tmp_path, "property-surefire-version", pom)

    profile = detect_check_profile(repo)

    assert profile.bootstrap_argv[1][-2] == (
        "-Dartifact=org.apache.maven.surefire:surefire-junit-platform:3.5.2"
    )
    assert profile.bootstrap_argv[2][-2] == (
        "-Dartifact=org.junit.platform:junit-platform-launcher:1.9.3"
    )


def test_maven_surefire_version_outside_allowlist_is_rejected(tmp_path: Path) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "<version>3.5.2</version>",
        "<version>3.5.1</version>",
    )
    repo = _repository_with_pom(tmp_path, "unsupported-surefire-version", pom)

    with pytest.raises(
        CheckPolicyError,
        match="maven-surefire-plugin version is not supported; expected 3.5.2",
    ):
        detect_check_profile(repo)


def test_maven_surefire_malicious_version_is_rejected(tmp_path: Path) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "<version>3.5.2</version>",
        "<version>3.5.2;touch-pwned</version>",
    )
    repo = _repository_with_pom(tmp_path, "malicious-surefire-version", pom)

    with pytest.raises(
        CheckPolicyError,
        match="maven-surefire-plugin version is invalid",
    ):
        detect_check_profile(repo)


def test_maven_junit_jupiter_version_is_required(tmp_path: Path) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "      <version>5.11.4</version>\n", ""
    )
    repo = _repository_with_pom(tmp_path, "missing-junit-version", pom)

    with pytest.raises(
        CheckPolicyError,
        match="org.junit.jupiter:junit-jupiter must declare one explicit version",
    ):
        detect_check_profile(repo)


def test_maven_junit_jupiter_property_version_is_resolved_for_bootstrap(
    tmp_path: Path,
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "    <maven.compiler.release>17</maven.compiler.release>\n",
        "    <maven.compiler.release>17</maven.compiler.release>\n"
        "    <junit.version>5.11.4</junit.version>\n",
    ).replace(
        "      <version>5.11.4</version>",
        "      <version>${junit.version}</version>",
    )
    repo = _repository_with_pom(tmp_path, "property-junit-version", pom)

    profile = detect_check_profile(repo)

    assert profile.bootstrap_argv[3][-2] == (
        "-Dartifact=org.junit.platform:junit-platform-launcher:1.11.4"
    )


def test_maven_junit_jupiter_version_outside_allowlist_is_rejected(
    tmp_path: Path,
) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "<version>5.11.4</version>",
        "<version>5.11.3</version>",
    )
    repo = _repository_with_pom(tmp_path, "unsupported-junit-version", pom)

    with pytest.raises(
        CheckPolicyError,
        match="JUnit Jupiter version is not supported; expected 5.11.4",
    ):
        detect_check_profile(repo)


def test_maven_junit_jupiter_malicious_version_is_rejected(tmp_path: Path) -> None:
    pom = MAVEN_POM.format(release="17").replace(
        "<version>5.11.4</version>",
        "<version>5.11.4;touch-pwned</version>",
    )
    repo = _repository_with_pom(tmp_path, "malicious-junit-version", pom)

    with pytest.raises(
        CheckPolicyError,
        match="org.junit.jupiter:junit-jupiter version is invalid",
    ):
        detect_check_profile(repo)


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"build.gradle": "plugins {}\n"}, "Gradle projects are not supported"),
        ({"pom.xml": MAVEN_POM.format(release="17")}, "mixed Python and Maven"),
        ({"module/pom.xml": MAVEN_POM.format(release="17")}, "single-module Maven"),
    ],
)
def test_rejects_unsupported_or_ambiguous_repository_layouts(
    tmp_path: Path, files: dict[str, str], message: str
) -> None:
    repo = tmp_path / "rejected-source"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_app.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _commit_repository(repo)

    with pytest.raises(CheckPolicyError, match=message):
        detect_check_profile(repo)


def test_rejects_maven_modules_and_unsupported_java_release(tmp_path: Path) -> None:
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "pom.xml").write_text(
        MAVEN_POM.format(release="17").replace(
            "</project>", "<modules><module>child</module></modules></project>"
        ),
        encoding="utf-8",
    )
    _commit_repository(modules)
    with pytest.raises(CheckPolicyError, match="multi-module Maven"):
        detect_check_profile(modules)

    unsupported = _maven_repository(tmp_path, release="8")
    with pytest.raises(CheckPolicyError, match="release must be 17 or 21"):
        detect_check_profile(unsupported)


def test_python_verify_uses_hardened_offline_container_and_exact_cleanup(
    tmp_path: Path,
) -> None:
    repo = _python_repository(tmp_path)
    source_head = _git(repo, "rev-parse", "HEAD")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo,
        command_runner=runner,
        temp_parent=tmp_path,
        phase_timeout_seconds=15,
    ).run()

    assert result.ok
    assert result.status == "passed"
    assert result.base_commit == source_head
    assert result.candidate_applied is False
    assert result.cleanup_ok
    assert len(result.phases) == 1
    phase = result.phases[0]
    assert phase.kind == "verify"
    assert phase.network == "none"
    assert phase.argv == result.profile.check_argv  # type: ignore[union-attr]

    assert [command[2] for command in runner.commands] == ["create", "start", "rm"]
    create = runner.commands[0]
    assert _option_values(create, "--network") == ["none"]
    assert "--read-only" in create
    assert _option_values(create, "--user") == ["10001:10001"]
    assert _option_values(create, "--cpus") == ["2"]
    assert _option_values(create, "--memory") == ["4g"]
    assert _option_values(create, "--memory-swap") == ["4g"]
    assert _option_values(create, "--pids-limit") == ["256"]
    assert _option_values(create, "--cap-drop") == ["ALL"]
    assert _option_values(create, "--security-opt") == ["no-new-privileges"]
    assert _option_values(create, "--log-driver") == ["none"]
    assert _option_values(create, "--tmpfs") == [
        "/tmp:rw,nosuid,nodev,size=256m,mode=1777"
    ]
    assert set(_option_values(create, "--env")) == {
        "HOME=/tmp/home",
        "CI=1",
        "NO_COLOR=1",
        "PYTHONDONTWRITEBYTECODE=1",
    }
    mounts = _mount_sources(create)
    assert set(mounts) == {"/workspace", "/dependencies"}
    assert Path(mounts["/workspace"]) != repo
    assert Path(mounts["/dependencies"]) != Path.home()
    assert not Path(mounts["/workspace"]).exists()
    assert "/var/run/docker.sock" not in " ".join(create)
    entrypoint = create.index("--entrypoint")
    assert create[entrypoint : entrypoint + 3] == (
        "--entrypoint",
        "python",
        "repo-agent-python:0.1",
    )
    assert create[entrypoint + 3 :] == (
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    )
    assert runner.commands[-1] == (
        "docker",
        "container",
        "rm",
        "--force",
        CONTAINER_IDS[0],
    )
    assert runner.timeouts[1] <= 15


def test_bootstrap_requires_authorization_before_any_container_is_created(
    tmp_path: Path,
) -> None:
    repo = _python_repository(tmp_path, requirements="requests==2.32.5\n")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path, allow_bootstrap=False
    ).run()

    assert result.status == "bootstrap_required"
    assert not result.ok
    assert result.profile is not None and result.profile.bootstrap_required
    assert result.phases == ()
    assert "allow_bootstrap=True" in (result.error or "")
    assert runner.commands == []


def test_authorized_bootstrap_is_networked_and_verify_is_fresh_and_offline(
    tmp_path: Path,
) -> None:
    repo = _python_repository(tmp_path, requirements="requests==2.32.5\n")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path, allow_bootstrap=True
    ).run()

    assert result.ok
    assert [phase.kind for phase in result.phases] == ["bootstrap", "verify"]
    assert [phase.network for phase in result.phases] == ["bridge", "none"]
    creates = [command for command in runner.commands if command[2] == "create"]
    assert len(creates) == 2
    assert [_option_values(command, "--network") for command in creates] == [
        ["bridge"],
        ["none"],
    ]
    assert all(
        "PYTHONPATH=/dependencies/python" in _option_values(command, "--env")
        for command in creates
    )
    mount_sets = [_mount_sources(command) for command in creates]
    bootstrap_workspace = mount_sets[0]["/workspace"]
    verify_workspace = mount_sets[1]["/workspace"]
    assert verify_workspace != bootstrap_workspace
    assert len({mounts["/dependencies"] for mounts in mount_sets}) == 1
    assert all(not Path(mounts["/workspace"]).exists() for mounts in mount_sets)
    assert all(not Path(mounts["/dependencies"]).exists() for mounts in mount_sets)
    assert [command[2] for command in runner.commands] == [
        "create",
        "start",
        "rm",
        "create",
        "start",
        "rm",
    ]


def test_candidate_patch_is_applied_only_to_fresh_verification_copy(
    tmp_path: Path,
) -> None:
    repo = _python_repository(tmp_path)
    before_head = _git(repo, "rev-parse", "HEAD")
    before_status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    before_content = (repo / "app.py").read_bytes()
    observed: list[tuple[Path, str]] = []

    def inspect_workspace(create: tuple[str, ...]) -> None:
        workspace = Path(_mount_sources(create)["/workspace"])
        observed.append(
            (workspace, (workspace / "app.py").read_text(encoding="utf-8"))
        )

    patch = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""
    result = CheckRunner(
        repo,
        candidate_patch=patch,
        command_runner=FakeDockerRunner(on_start=inspect_workspace),
        temp_parent=tmp_path,
    ).run()

    assert result.ok
    assert result.candidate_applied
    assert len(observed) == 1
    workspace, content = observed[0]
    assert workspace != repo
    assert content == "VALUE = 2\n"
    assert not workspace.exists()
    assert _git(repo, "rev-parse", "HEAD") == before_head
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == before_status
    assert (repo / "app.py").read_bytes() == before_content


def test_maven_bootstrap_and_verify_use_only_fixed_commands(tmp_path: Path) -> None:
    repo = _maven_repository(tmp_path, release="17")
    runner = FakeDockerRunner()

    blocked = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path
    ).run("maven-test")
    assert blocked.status == "bootstrap_required"
    assert runner.commands == []

    allowed_runner = FakeDockerRunner()
    result = CheckRunner(
        repo,
        command_runner=allowed_runner,
        temp_parent=tmp_path,
        allow_bootstrap=True,
    ).run("maven-test")

    assert result.ok
    assert [phase.kind for phase in result.phases] == [
        "bootstrap",
        "bootstrap",
        "bootstrap",
        "bootstrap",
        "verify",
    ]
    assert [phase.network for phase in result.phases] == [
        "bridge",
        "bridge",
        "bridge",
        "bridge",
        "none",
    ]
    assert result.phases[0].argv[-2:] == (
        "-DskipTests",
        "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:go-offline",
    )
    assert result.phases[1].argv[-3:] == (
        "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
        "-Dartifact=org.apache.maven.surefire:surefire-junit-platform:3.5.2",
        "-Dtransitive=true",
    )
    assert result.phases[2].argv[-3:] == (
        "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
        "-Dartifact=org.junit.platform:junit-platform-launcher:1.9.3",
        "-Dtransitive=true",
    )
    assert result.phases[3].argv[-3:] == (
        "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
        "-Dartifact=org.junit.platform:junit-platform-launcher:1.11.4",
        "-Dtransitive=true",
    )
    assert "--offline" in result.phases[4].argv
    assert result.phases[4].argv[-2:] == (
        "process-test-classes",
        "org.apache.maven.plugins:maven-surefire-plugin:3.5.2:test",
    )
    creates = [command for command in allowed_runner.commands if command[2] == "create"]
    assert [command[command.index("--entrypoint") + 1] for command in creates] == [
        "mvn",
        "mvn",
        "mvn",
        "mvn",
        "mvn",
    ]
    assert all("repo-agent-maven:0.1" in command for command in creates)


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
            "java.net.SocketException: Connection reset"
        ),
        (
            "Could not transfer metadata example:fixture/maven-metadata.xml "
            "from/to central: status code: 503, reason phrase: Service Unavailable"
        ),
        (
            "Could not transfer artifact example:fixture:jar:1 from/to central: "
            "Premature end of Content-Length delimited message body "
            "(expected: 246,918; received: 229,376)"
        ),
    ],
)
def test_maven_bootstrap_retries_once_for_transient_transfer_failure(
    tmp_path: Path, failure: str
) -> None:
    runner = FakeDockerRunner(
        starts=[CommandOutcome(1, failure), CommandOutcome(0, "retry succeeded")]
    )

    result = CheckRunner(
        _maven_repository(tmp_path),
        command_runner=runner,
        temp_parent=tmp_path,
        allow_bootstrap=True,
    ).run("maven-test")

    assert result.ok
    assert len(result.phases) == 5
    assert result.phases[0].status == "passed"
    assert failure in result.phases[0].output
    assert "retry succeeded" in result.phases[0].output
    creates = [command for command in runner.commands if command[2] == "create"]
    assert len(creates) == 6
    assert _mount_sources(creates[0]) == _mount_sources(creates[1])
    assert (
        creates[0][creates[0].index("repo-agent-maven:0.1") :]
        == creates[1][creates[1].index("repo-agent-maven:0.1") :]
    )


def test_maven_bootstrap_stops_after_one_transient_retry(tmp_path: Path) -> None:
    first_failure = (
        "Could not transfer artifact example:fixture:jar:1 from/to central: "
        "Remote host terminated the handshake: SSL peer shut down incorrectly"
    )
    second_failure = (
        "Could not transfer artifact example:fixture:jar:1 from/to central: "
        "java.net.SocketException: Connection reset"
    )
    runner = FakeDockerRunner(
        starts=[CommandOutcome(1, first_failure), CommandOutcome(1, second_failure)]
    )

    result = CheckRunner(
        _maven_repository(tmp_path),
        command_runner=runner,
        temp_parent=tmp_path,
        allow_bootstrap=True,
    ).run("maven-test")

    assert result.status == "failed"
    assert len(result.phases) == 1
    assert first_failure in result.phases[0].output
    assert second_failure in result.phases[0].output
    assert len([command for command in runner.commands if command[2] == "create"]) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "[ERROR] COMPILATION ERROR: cannot find symbol",
        "[ERROR] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0",
        "Could not resolve artifact example:missing:jar:1",
        (
            "Could not transfer artifact example:fixture:jar:1 from/to central: "
            "PKIX path building failed: unable to find valid certification path"
        ),
        "Remote host terminated the handshake: SSL peer shut down incorrectly",
    ],
)
def test_maven_bootstrap_does_not_retry_deterministic_failure(
    tmp_path: Path, failure: str
) -> None:
    runner = FakeDockerRunner(starts=[CommandOutcome(1, failure)])

    result = CheckRunner(
        _maven_repository(tmp_path),
        command_runner=runner,
        temp_parent=tmp_path,
        allow_bootstrap=True,
    ).run("maven-test")

    assert result.status == "failed"
    assert len(result.phases) == 1
    assert len([command for command in runner.commands if command[2] == "create"]) == 1


def test_wrong_check_id_is_denied_without_starting_docker(tmp_path: Path) -> None:
    runner = FakeDockerRunner()

    result = CheckRunner(
        _python_repository(tmp_path), command_runner=runner, temp_parent=tmp_path
    ).run("maven-test")

    assert result.status == "policy_denied"
    assert "does not match detected profile" in (result.error or "")
    assert runner.commands == []


def test_timeout_is_structured_truncated_and_removes_exact_container(
    tmp_path: Path,
) -> None:
    runner = FakeDockerRunner(
        starts=[CommandOutcome(None, "x" * 4096, timed_out=True)]
    )

    result = CheckRunner(
        _python_repository(tmp_path),
        command_runner=runner,
        temp_parent=tmp_path,
        phase_timeout_seconds=0.25,
        # This case exercises a container phase timeout, not host Git speed.
        # Leave enough headroom for a loaded Windows filesystem.
        total_timeout_seconds=10,
        max_output_bytes=1024,
    ).run()

    assert result.status == "timed_out"
    assert not result.ok
    assert result.cleanup_ok
    assert len(result.phases) == 1
    phase = result.phases[0]
    assert phase.status == "timed_out"
    assert phase.exit_code is None
    assert phase.output == "x" * 1024
    assert phase.truncated
    assert "timed out" in (phase.error or "")
    assert [command[2] for command in runner.commands] == ["create", "start", "rm"]
    assert runner.commands[-1][-1] == CONTAINER_IDS[0]
    assert 0 < runner.timeouts[1] <= 0.25
    assert runner.output_limits == [1024, 1024, 1024]


@pytest.mark.parametrize("create_timeout", [False, True])
def test_malformed_or_timed_out_create_recovers_owned_container_by_exact_id(
    tmp_path: Path, create_timeout: bool
) -> None:
    create = CommandOutcome(
        None if create_timeout else 0,
        "create did not return an id\n",
        timed_out=create_timeout,
    )
    runner = FakeDockerRunner(creates=[create], inspect_owned=True)

    result = CheckRunner(
        _python_repository(tmp_path), command_runner=runner, temp_parent=tmp_path
    ).run()

    assert result.status == "setup_error"
    assert result.cleanup_ok
    assert [command[2] for command in runner.commands] == ["create", "inspect", "rm"]
    inspect = runner.commands[1]
    created_name = runner.commands[0][runner.commands[0].index("--name") + 1]
    assert inspect[-1] == created_name
    assert runner.commands[2][-1] == CONTAINER_IDS[0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"phase_timeout_seconds": 301}, "phase_timeout_seconds"),
        ({"total_timeout_seconds": 1201}, "total_timeout_seconds"),
        ({"max_output_bytes": 1023}, "max_output_bytes"),
        ({"max_output_bytes": MAX_CHECK_OUTPUT_BYTES + 1}, "max_output_bytes"),
        ({"allow_bootstrap": 1}, "allow_bootstrap"),
    ],
)
def test_runner_rejects_out_of_policy_budgets(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CheckRunner(_python_repository(tmp_path), temp_parent=tmp_path, **kwargs)


def test_invalid_candidate_patch_is_rejected_before_docker(tmp_path: Path) -> None:
    runner = FakeDockerRunner()
    with pytest.raises(PatchValidationError):
        CheckRunner(
            _python_repository(tmp_path),
            candidate_patch="not a unified diff",
            command_runner=runner,
            temp_parent=tmp_path,
        )
    assert runner.commands == []


def test_dirty_source_is_denied_and_never_mounted_or_modified(tmp_path: Path) -> None:
    repo = _python_repository(tmp_path)
    (repo / "app.py").write_text("VALUE = 99\n", encoding="utf-8")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path
    ).run()

    assert result.status == "policy_denied"
    assert "clean worktree" in (result.error or "")
    assert runner.commands == []
    assert (repo / "app.py").read_text(encoding="utf-8") == "VALUE = 99\n"


def test_committed_sensitive_path_is_denied_before_docker(tmp_path: Path) -> None:
    repo = _python_repository(tmp_path)
    (repo / ".env").write_text("TOKEN=must-not-enter-check\n", encoding="utf-8")
    _git(repo, "add", ".env")
    _git(repo, "commit", "--quiet", "-m", "unsafe fixture")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path
    ).run()

    assert result.status == "policy_denied"
    assert "unsafe tracked path" in (result.error or "")
    assert runner.commands == []
    assert (repo / ".env").read_text(encoding="utf-8") == (
        "TOKEN=must-not-enter-check\n"
    )


def test_git_lfs_attributes_are_denied_before_docker(tmp_path: Path) -> None:
    repo = _python_repository(tmp_path)
    (repo / ".gitattributes").write_text(
        "*.bin filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "--quiet", "-m", "lfs fixture")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path
    ).run()

    assert result.status == "policy_denied"
    assert "Git LFS" in (result.error or "")
    assert runner.commands == []


def test_maven_config_is_denied_before_docker(tmp_path: Path) -> None:
    repo = _maven_repository(tmp_path)
    config = repo / ".mvn" / "maven.config"
    config.parent.mkdir()
    config.write_text("--fail-never\n", encoding="utf-8")
    _git(repo, "add", ".mvn/maven.config")
    _git(repo, "commit", "--quiet", "-m", "unsafe Maven config")
    docker = FakeDockerRunner()

    result = CheckRunner(
        repo,
        allow_bootstrap=True,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert ".mvn" in (result.error or "")
    assert docker.commands == []


def test_gitlink_is_denied_before_docker(tmp_path: Path) -> None:
    repo = _python_repository(tmp_path)
    nested = repo / "vendor" / "lib"
    nested.mkdir(parents=True)
    _git(nested, "init", "--quiet")
    _git(nested, "config", "user.name", "Nested Check Test")
    _git(nested, "config", "user.email", "nested@example.invalid")
    (nested / "README.md").write_text("nested\n", encoding="utf-8")
    _git(nested, "add", "README.md")
    _git(nested, "commit", "--quiet", "-m", "nested fixture")
    _git(repo, "add", "vendor/lib")
    _git(repo, "commit", "--quiet", "-m", "gitlink fixture")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path
    ).run()

    assert result.status == "policy_denied"
    assert "submodules are not supported" in (result.error or "")
    assert runner.commands == []


def test_symlink_is_denied_before_docker_when_supported(tmp_path: Path) -> None:
    repo = _python_repository(tmp_path)
    link = repo / "linked.py"
    try:
        link.symlink_to("app.py")
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    _git(repo, "add", "linked.py")
    _git(repo, "commit", "--quiet", "-m", "symlink fixture")
    runner = FakeDockerRunner()

    result = CheckRunner(
        repo, command_runner=runner, temp_parent=tmp_path
    ).run()

    assert result.status == "policy_denied"
    assert "symbolic links are not supported" in (result.error or "")
    assert runner.commands == []


def test_nonzero_verify_exit_is_failed_and_container_is_removed(tmp_path: Path) -> None:
    runner = FakeDockerRunner(starts=[CommandOutcome(7, "tests failed\n")])

    result = CheckRunner(
        _python_repository(tmp_path), command_runner=runner, temp_parent=tmp_path
    ).run()

    assert result.status == "failed"
    assert result.phases[0].status == "failed"
    assert result.phases[0].exit_code == 7
    assert result.phases[0].output == "tests failed\n"
    assert result.cleanup_ok
    assert [command[2] for command in runner.commands] == ["create", "start", "rm"]
    assert runner.commands[-1][-1] == CONTAINER_IDS[0]


def test_timed_out_create_with_repeated_absence_keeps_cleanup_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        checks_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    absent = CommandOutcome(1, "Error response from daemon: No such container\n")
    docker = FakeDockerRunner(
        creates=[CommandOutcome(None, "", timed_out=True)],
        inspects=[absent] * 8,
    )
    check_runner = CheckRunner(
        _python_repository(tmp_path),
        command_runner=docker,
        control_timeout_seconds=0.2,
        temp_parent=tmp_path,
    )
    monkeypatch.setattr(check_runner, "_host_timeout", lambda _started: 30.0)

    try:
        result = check_runner.run()

        assert result.status == "cleanup_error"
        assert result.cleanup_ok is False
        assert check_runner._root is not None
        assert check_runner._root.exists()
        create = next(command for command in docker.commands if command[2] == "create")
        container_name = create[create.index("--name") + 1]
        assert container_name in check_runner._container_names
    finally:
        docker.inspects.clear()
        docker.inspect_owned = True
        check_runner._cleanup_deadline = None
        check_runner.close()


def test_keyboard_interrupt_still_removes_exact_container_and_temporary_root(
    tmp_path: Path,
) -> None:
    docker = FakeDockerRunner(start_exception=KeyboardInterrupt())
    check_runner = CheckRunner(
        _python_repository(tmp_path), command_runner=docker, temp_parent=tmp_path
    )
    operations_after_interrupt: list[str] = []
    cleanup_target: str | None = None
    root_after_interrupt: Path | None = None

    try:
        with pytest.raises(KeyboardInterrupt):
            check_runner.run()
        operations_after_interrupt = [command[2] for command in docker.commands]
        cleanup_target = docker.commands[-1][-1]
        root_after_interrupt = check_runner._root
    finally:
        check_runner.close()

    assert operations_after_interrupt == ["create", "start", "rm"]
    assert cleanup_target == CONTAINER_IDS[0]
    assert root_after_interrupt is None


def test_keyboard_interrupt_is_preserved_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        checks_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    docker = FakeDockerRunner(start_exception=KeyboardInterrupt())
    cleanup_should_fail = [True]

    def fail_first_cleanup(
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        command = tuple(argv)
        if command[2] == "rm" and cleanup_should_fail[0]:
            docker.commands.append(command)
            docker.timeouts.append(timeout_seconds)
            docker.output_limits.append(max_output_bytes)
            return CommandOutcome(1, "forced cleanup failure\n")
        return docker(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    check_runner = CheckRunner(
        _python_repository(tmp_path),
        command_runner=fail_first_cleanup,
        control_timeout_seconds=1,
        temp_parent=tmp_path,
    )

    try:
        with pytest.raises(KeyboardInterrupt):
            check_runner.run()
    finally:
        cleanup_should_fail[0] = False
        check_runner._cleanup_deadline = None
        check_runner.close()


def test_candidate_dependencies_are_profiled_before_bootstrap_authorization(
    tmp_path: Path,
) -> None:
    patch = """\
diff --git a/pyproject.toml b/pyproject.toml
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -5,4 +5,4 @@ build-backend = "setuptools.build_meta"
 [project]
 name = "check-fixture"
 version = "0.1.0"
-dependencies = []
+dependencies = ["requests==2.32.5"]
"""
    docker = FakeDockerRunner()

    result = CheckRunner(
        _python_repository(tmp_path),
        candidate_patch=patch,
        allow_bootstrap=False,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "bootstrap_required"
    assert result.profile is not None and result.profile.bootstrap_required
    assert result.candidate_applied
    assert docker.commands == []


def test_candidate_adding_gradle_marker_is_denied_before_docker(tmp_path: Path) -> None:
    patch = """\
diff --git a/build.gradle b/build.gradle
new file mode 100644
--- /dev/null
+++ b/build.gradle
@@ -0,0 +1 @@
+plugins {}
"""
    docker = FakeDockerRunner()

    result = CheckRunner(
        _python_repository(tmp_path),
        candidate_patch=patch,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert "Gradle projects are not supported" in (result.error or "")
    assert result.candidate_applied
    assert docker.commands == []


def test_candidate_adding_nested_pom_is_denied_before_docker(tmp_path: Path) -> None:
    nested_pom = """\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>example</groupId>
  <artifactId>nested</artifactId>
  <version>1.0.0</version>
</project>
"""
    patch = (
        "diff --git a/module/pom.xml b/module/pom.xml\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/module/pom.xml\n"
        f"@@ -0,0 +1,{len(nested_pom.splitlines())} @@\n"
        + "".join(f"+{line}\n" for line in nested_pom.splitlines())
    )
    docker = FakeDockerRunner()

    result = CheckRunner(
        _maven_repository(tmp_path),
        candidate_patch=patch,
        command_runner=docker,
        temp_parent=tmp_path,
    ).run()

    assert result.status == "policy_denied"
    assert "single-module Maven" in (result.error or "")
    assert result.candidate_applied
    assert docker.commands == []


def test_maven_compiler_plugin_release_is_policy_enforced(tmp_path: Path) -> None:
    unsupported = _maven_plugin_repository(tmp_path, "plugin-release-8", "8")
    with pytest.raises(CheckPolicyError, match="release must be 17 or 21"):
        detect_check_profile(unsupported)

    supported = _maven_plugin_repository(tmp_path, "plugin-release-17", "17")
    profile = detect_check_profile(supported)

    assert profile.id == "maven-test"
    assert profile.java_release == 17


def test_create_and_start_share_one_phase_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    docker = FakeDockerRunner()

    def advance_after_create(
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        outcome = docker(
            argv,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        if tuple(argv)[2] == "create":
            clock[0] += 4.0
        return outcome

    result = CheckRunner(
        _python_repository(tmp_path),
        command_runner=advance_after_create,
        temp_parent=tmp_path,
        phase_timeout_seconds=10,
    ).run()

    assert result.ok
    start_index = next(
        index for index, command in enumerate(docker.commands) if command[2] == "start"
    )
    assert 0 < docker.timeouts[start_index] <= 6


def test_host_git_timeouts_are_capped_by_remaining_run_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _python_repository(tmp_path)
    started = 100.0
    clock = [started]
    total_timeout = 10.0
    original_run = subprocess.run
    observed: list[tuple[float | None, float]] = []
    check_runner = CheckRunner(
        repo,
        command_runner=FakeDockerRunner(),
        temp_parent=tmp_path,
        phase_timeout_seconds=10,
        total_timeout_seconds=total_timeout,
    )

    def advance_after_git(*args, **kwargs):
        remaining = total_timeout - (clock[0] - started)
        observed.append((kwargs.get("timeout"), remaining))
        completed = original_run(*args, **kwargs)
        clock[0] += 0.5
        return completed

    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(checks_module.subprocess, "run", advance_after_git)

    result = check_runner.run()

    assert result.ok
    assert observed
    assert all(
        timeout is not None and 0 < timeout <= remaining
        for timeout, remaining in observed
    )


def test_fresh_copy_stops_host_git_after_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _python_repository(tmp_path)
    started = 200.0
    clock = [started]
    deadline = started + 1.0
    original_run = subprocess.run
    late_git_commands: list[tuple[str, ...]] = []

    def expire_after_fresh_clone(*args, **kwargs):
        command = tuple(args[0])
        if clock[0] >= deadline:
            late_git_commands.append(command)
        completed = original_run(*args, **kwargs)
        if "clone" in command and Path(command[-1]).name == "verify":
            clock[0] = deadline + 1.0
        return completed

    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(checks_module.subprocess, "run", expire_after_fresh_clone)
    docker = FakeDockerRunner()

    result = CheckRunner(
        repo,
        command_runner=docker,
        temp_parent=tmp_path,
        phase_timeout_seconds=1,
        total_timeout_seconds=1,
    ).run()

    assert result.status == "timed_out"
    assert late_git_commands == []
    assert docker.commands == []


def test_successful_verify_cannot_pass_after_total_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [300.0]
    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])

    def finish_after_deadline(_create_command: tuple[str, ...]) -> None:
        clock[0] = 311.0

    docker = FakeDockerRunner(on_start=finish_after_deadline)
    result = CheckRunner(
        _python_repository(tmp_path),
        command_runner=docker,
        temp_parent=tmp_path,
        phase_timeout_seconds=10,
        total_timeout_seconds=10,
    ).run()

    assert result.status == "timed_out"
    assert not result.ok
    assert result.cleanup_ok


def test_timed_out_create_recovers_container_appearing_within_control_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [400.0]
    absent = CommandOutcome(1, "Error response from daemon: No such container\n")
    docker = FakeDockerRunner(
        creates=[CommandOutcome(None, "", timed_out=True)],
        inspects=[absent] * 8,
        inspect_owned=True,
    )

    def advance_control_clock(
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        outcome = docker(
            argv,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        if tuple(argv)[2] == "inspect":
            clock[0] += 1.0
        return outcome

    monkeypatch.setattr(checks_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        checks_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    check_runner = CheckRunner(
        _python_repository(tmp_path),
        command_runner=advance_control_clock,
        temp_parent=tmp_path,
    )

    try:
        result = check_runner.run()

        assert result.status == "setup_error"
        assert result.cleanup_ok
        operations = [command[2] for command in docker.commands]
        assert operations.count("inspect") >= 9
        assert operations[-1] == "rm"
        assert docker.commands[-1][-1] == CONTAINER_IDS[0]
    finally:
        docker.inspects.clear()
        docker.inspect_owned = True
        check_runner.close()
