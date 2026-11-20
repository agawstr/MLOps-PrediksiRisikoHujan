FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

RUN set -eux; \
    grep -rl "yarnpkg" /etc/apt/ 2>/dev/null | xargs -r rm -f; \
    rm -f /usr/share/keyrings/yarn*.gpg /etc/apt/trusted.gpg.d/yarn*.gpg; \
    apt-get update || true