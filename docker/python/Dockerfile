FROM python:3.11.16-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 repo-agent \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/repo-agent repo-agent \
    && mkdir --parents /workspace /cache \
    && chown --recursive 10001:10001 /workspace /cache /home/repo-agent

RUN python -m pip install --no-cache-dir --disable-pip-version-check pytest==9.1.1

ENV HOME=/home/repo-agent \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
USER 10001:10001

ENTRYPOINT []
CMD ["python", "--version"]
