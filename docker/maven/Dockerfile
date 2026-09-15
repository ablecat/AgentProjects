FROM maven:3.9.16-eclipse-temurin-21-noble@sha256:8f6ac126f7810bb5549c4cd122d2bf0e9cda5bdeb0838aa928f09e779fd8bef8

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install --yes --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 repo-agent \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/repo-agent repo-agent \
    && mkdir --parents /workspace /cache /home/repo-agent/.m2 \
    && chown --recursive 10001:10001 /workspace /cache /home/repo-agent

RUN mkdir --parents /cache/m2 \
    && chown 10001:10001 /cache/m2

ENV HOME=/home/repo-agent \
    MAVEN_CONFIG=/home/repo-agent/.m2

WORKDIR /workspace
USER 10001:10001

ENTRYPOINT []
CMD ["mvn", "--version"]
