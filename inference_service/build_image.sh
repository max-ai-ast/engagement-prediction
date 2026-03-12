#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="inference-service"
DOCKERFILE_PATH="inference_service/Dockerfile"

if [[ "${1:-}" == "--gpu" ]]; then
  DOCKERFILE_PATH="inference_service/Dockerfile.gpu"
  IMAGE_NAME=${IMAGE_NAME}-gpu
  shift
fi

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [--gpu]"
  exit 0
fi

# 1. Get git commit SHA
GIT_SHA=$(git rev-parse HEAD)
SHORT_SHA=${GIT_SHA:0:12}

echo "Git SHA: $GIT_SHA"

# 2. Build docker image
docker build \
  --build-arg GIT_SHA="$GIT_SHA" \
  -f "$DOCKERFILE_PATH" \
  -t "${IMAGE_NAME}:git-${SHORT_SHA}" \
  inference_service

# 3. Extract local image digest
IMAGE_DIGEST=$(docker inspect \
  --format='{{index .RepoDigests 0}}' \
  "${IMAGE_NAME}:git-${SHORT_SHA}" 2>/dev/null || true)

# If no repo digest exists yet (no registry push), fall back to image ID
if [[ -z "$IMAGE_DIGEST" ]]; then
  IMAGE_DIGEST=$(docker inspect \
    --format='{{.Id}}' \
    "${IMAGE_NAME}:git-${SHORT_SHA}")
fi

echo ""
echo "Built image:"
echo "  Tag:    ${IMAGE_NAME}:git-${SHORT_SHA}"
echo "  Digest: $IMAGE_DIGEST"
