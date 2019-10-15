#!/usr/bin/env bash

DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollo"

REPO="${DOCKER_REGISTRY}/${DOCKER_USER}/spark_submitter"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"
# Go to spark_submitter root.
cd $( dirname "${BASH_SOURCE[0]}" )/..

set -e
set -x

# Prepare resource.
protoc --python_out=. *.proto
cp ~/.kube/config deploy/kube.config

# Build.
docker build -t ${IMAGE} --network host -f deploy/Dockerfile .
