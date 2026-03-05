#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="inference-service"

# 1. Get git commit SHA
GIT_SHA=$(git rev-parse HEAD)
SHORT_SHA=${GIT_SHA:0:12}

echo "Git SHA: $GIT_SHA"

# 2. Build docker image
docker build \
  --build-arg GIT_SHA="$GIT_SHA" \
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

echo "DOCKER_IMAGE_DIGEST=$IMAGE_DIGEST" > .docker_image.env
echo "DOCKER_IMAGE_TAG=${IMAGE_NAME}:git-${SHORT_SHA}" >> .docker_image.env

echo ""
echo "Built image:"
echo "  Tag:    ${IMAGE_NAME}:git-${SHORT_SHA}"
echo "  Digest: $IMAGE_DIGEST"