#!/usr/bin/env bash

REPO="hub.baidubce.com/apollo/warehouse"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"
# Go to apollo-fuel root.
cd $( dirname "${BASH_SOURCE[0]}" )/../../../..

set -e
set -x

# Build your local apollo.
bash ../apollo/apollo_docker.sh build_py
cp -r ../apollo/py_proto ./

docker build -t ${IMAGE} --network host -f apps/k8s/warehouse/deploy/Dockerfile .

# Login.
DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollo"
DOCKER_PASSWORD=""
# Please provide password if you want to login automatically.
if [ ! -z "${DOCKER_PASSWORD}" ]; then
  docker login -u ${DOCKER_USER} -p ${DOCKER_PASSWORD} ${DOCKER_REGISTRY}
fi

if [ "$1" = "push" ]; then
  docker push ${IMAGE}
else
  echo "Now you can push the images with: docker push ${IMAGE}"
fi
