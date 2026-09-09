# Repo Maintainer Agent

This repository currently contains a base-image-pinned Docker sandbox foundation for a Python and Maven code-maintenance agent. Final local image IDs are recorded after each successful build.

## Prerequisites

- Windows with WSL2 and Docker Desktop using the `desktop-linux` context
- PowerShell 5.1 or later
- At least 4 GB of memory available to Docker

Docker Desktop remains opt-in at login. The scripts do not create or edit `.wslconfig`, registry mirrors, proxies, or global Docker cleanup settings.

## Build and verify

```powershell
./scripts/start-docker.ps1
./scripts/build-sandboxes.ps1
./scripts/verify-docker.ps1
```

The two sandbox images are:

- `repo-agent-python:0.1`: Python 3.11, pytest, Git, and ripgrep
- `repo-agent-maven:0.1`: Eclipse Temurin JDK 21, Maven 3.9, Git, and ripgrep

Both images run as UID/GID `10001` by default. The verification script exercises them with no network, a read-only root filesystem, dropped Linux capabilities, `no-new-privileges`, CPU/memory/PID limits, and an isolated temporary workspace.

Existing Docker Desktop proxy and registry-mirror settings are reported but never changed. The script labels every temporary container and removes only containers and files created by its own run; it never invokes a global prune operation.

Every agent-image verification container uses the same isolation contract:

```text
--init --pull never --restart no
--network none --read-only --user 10001:10001
--cpus 2 --memory 4g --memory-swap 4g --pids-limit 256
--cap-drop ALL --security-opt no-new-privileges
--tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777
```

Only the run-specific workspace and dependency cache are writable bind mounts. The original repository, host home directory, SSH configuration, model credentials, and Docker socket are never mounted.

## Base image locks

| Sandbox | Immutable base image |
|---|---|
| Python | `python:3.11.16-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84` |
| Maven | `maven:3.9.16-eclipse-temurin-21-noble@sha256:8f6ac126f7810bb5549c4cd122d2bf0e9cda5bdeb0838aa928f09e779fd8bef8` |

The final locally built image IDs are recorded in `docker/image-lock.json` by `build-sandboxes.ps1` and checked by `verify-docker.ps1`.

The lock records the exact local build artifacts, while the Dockerfiles pin the base-image digests. Packages installed from live Debian, Ubuntu, and PyPI indexes can still change in a future rebuild; once CI publishing is introduced, CI and local runs should consume the published sandbox images by their final registry digests rather than rebuilding them independently.
