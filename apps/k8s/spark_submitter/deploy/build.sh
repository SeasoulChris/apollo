#!/usr/bin/env bash

DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollo"

REPO="${DOCKER_REGISTRY}/${DOCKER_USER}/spark_submitter"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"
# Go to fuel root.
cd $( dirname "${BASH_SOURCE[0]}" )/../../../..

set -e
set -x

# Prepare resource.
protoc --python_out=./ apps/k8s/spark_submitter/*.proto
cp ~/.kube/config ./kube.config

# Build.
docker build -t ${IMAGE} --network host -f apps/k8s/spark_submitter/deploy/Dockerfile .
