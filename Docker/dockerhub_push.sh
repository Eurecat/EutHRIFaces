#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 --env-file <path> --image <name:tag>"
}

normalize_bool() {
    local value="${1,,}"
    case "$value" in
        1|true|yes|y|on)
            echo "true"
            ;;
        *)
            echo "false"
            ;;
    esac
}

ENV_FILE=""
IMAGE_NAME=""

# Keep runtime override (for example, from --push-after-build) even if ENV_FILE
# defines DOCKERHUB_PUSH_AFTER_BUILD=false.
PUSH_AFTER_BUILD_OVERRIDE="${DOCKERHUB_PUSH_AFTER_BUILD:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "[WARNING] dockerhub_push: unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$ENV_FILE" ] || [ -z "$IMAGE_NAME" ]; then
    usage
    exit 1
fi

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [ -n "$PUSH_AFTER_BUILD_OVERRIDE" ]; then
    DOCKERHUB_PUSH_AFTER_BUILD="$PUSH_AFTER_BUILD_OVERRIDE"
fi

DOCKERHUB_NAMESPACE="${DOCKERHUB_NAMESPACE:-${DOCKERHUB_ORG:-}}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-}"
DOCKERHUB_TOKEN="${DOCKERHUB_TOKEN:-}"
DOCKERHUB_REPO_PREFIX="${DOCKERHUB_REPO_PREFIX:-}"
DOCKERHUB_PUSH_AFTER_BUILD="$(normalize_bool "${DOCKERHUB_PUSH_AFTER_BUILD:-false}")"

if [ "$DOCKERHUB_PUSH_AFTER_BUILD" != "true" ]; then
    exit 0
fi

if [ -z "$DOCKERHUB_NAMESPACE" ]; then
    echo "[WARNING] dockerhub_push: DOCKERHUB_NAMESPACE/DOCKERHUB_ORG missing. Skipping push for $IMAGE_NAME"
    exit 0
fi

if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
    echo "[WARNING] dockerhub_push: DOCKERHUB_USERNAME/DOCKERHUB_TOKEN missing. Skipping push for $IMAGE_NAME"
    exit 0
fi

if ! printf '%s' "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin >/dev/null 2>&1; then
    echo "[WARNING] dockerhub_push: docker login failed. Skipping push for $IMAGE_NAME"
    exit 0
fi

image_repo="${IMAGE_NAME%%:*}"
image_tag="${IMAGE_NAME##*:}"
remote_image="${DOCKERHUB_NAMESPACE}/${DOCKERHUB_REPO_PREFIX}${image_repo}:${image_tag}"

echo "[INFO] dockerhub_push: tagging $IMAGE_NAME as $remote_image"
docker tag "$IMAGE_NAME" "$remote_image"
echo "[INFO] dockerhub_push: pushing $remote_image"
docker push "$remote_image"
echo "[SUCCESS] dockerhub_push: pushed $remote_image"
