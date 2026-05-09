#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
    echo "error: 'docker' not found in PATH" >&2
    exit 127
fi
if ! docker info >/dev/null 2>&1; then
    echo "error: docker daemon not reachable" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="papergym-accumulator:latest"

export DOCKER_BUILDKIT=1
docker build -t "$TAG" -f "$REPO_ROOT/docker/Dockerfile" "$@" "$REPO_ROOT"

docker images --filter=reference='papergym-accumulator*' \
    --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
