#!/usr/bin/env bash

DOCKER_REGISTRY="registry.bce.baidu.com"
REPO="${DOCKER_REGISTRY}/a7e094e2914d424caa523a201e57995b/apollo-fuel-bae-proxy"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"

cd $( dirname "${BASH_SOURCE[0]}" )
set -e

cp ~/.kube/config ./kube.config
docker build -t ${IMAGE} --network host .
