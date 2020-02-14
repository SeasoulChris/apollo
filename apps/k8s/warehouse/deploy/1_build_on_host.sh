#!/usr/bin/env bash

DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollo"

REPO="${DOCKER_REGISTRY}/${DOCKER_USER}/warehouse"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"

cd "$( dirname "${BASH_SOURCE[0]}" )"
set -ex

# Build.
docker build -t ${IMAGE} --network host .

# Deploy.
sed -i "s|image: ${REPO}.*|image: ${IMAGE}|g" deploy.yaml
docker push ${IMAGE}
kubectl apply -f deploy.yaml
