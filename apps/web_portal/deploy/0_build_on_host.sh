#!/usr/bin/env bash

DOCKER_REGISTRY="registry.bce.baidu.com"
REPO="${DOCKER_REGISTRY}/9ced775163e5481b8b8ac99c7c14fe27/apollo-fuel-bae-proxy"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"

cd $( dirname "${BASH_SOURCE[0]}" )
set -e

cp ~/.kube/config ./kube.config
docker build -t ${IMAGE} --network host .
