#!/usr/bin/env bash

CONTAINER="fuel_${USER}"

cd "$( dirname "${BASH_SOURCE[0]}" )"
set -ex

# Build binary.
docker exec -u ${USER} ${CONTAINER} bazel build //apps/k8s/warehouse:index
docker exec -u ${USER} ${CONTAINER} \
    cp -f /fuel/bazel-bin/apps/k8s/warehouse/index.zip /fuel/apps/k8s/warehouse/deploy

# Build image.
REPO="hub.baidubce.com/apollofuel/warehouse"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"
docker build -t ${IMAGE} --network host .

# Deploy.
sed -i "s|image: ${REPO}.*|image: ${IMAGE}|g" deploy.yaml
docker push ${IMAGE}
kubectl apply -f deploy.yaml
