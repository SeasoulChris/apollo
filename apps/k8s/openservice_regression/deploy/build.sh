#!/usr/bin/env bash

CONTAINER="fuel"

cd "$( dirname "${BASH_SOURCE[0]}" )"
set -ex

# Build binary.
docker exec -u ${USER} ${CONTAINER} bazel build //apps/k8s/openservice_regression:openservice_regression
docker exec -u ${USER} ${CONTAINER} \
	    cp -f /fuel/bazel-bin/apps/k8s/openservice_regression/openservice_regression.zip /fuel/apps/k8s/openservice_regression/deploy

# Build image.
REPO="hub.baidubce.com/apollofuel/openservice_regression"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"
docker build -t ${IMAGE} --network host .

docker push ${IMAGE}
sed -i "s|image: ${REPO}.*|image: ${IMAGE}|g" deploy.yaml
kubectl apply -f deploy.yaml
