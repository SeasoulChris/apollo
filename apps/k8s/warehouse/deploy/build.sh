#!/usr/bin/env bash

DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollo"

REPO="${DOCKER_REGISTRY}/${DOCKER_USER}/warehouse"
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

# Please provide credential if you want to login automatically.
DOCKER_PASSWORD=""
if [ ! -z "${DOCKER_PASSWORD}" ]; then
  docker login -u ${DOCKER_USER} -p ${DOCKER_PASSWORD} ${DOCKER_REGISTRY}
fi

if [ "$1" = "push" ]; then
  docker push ${IMAGE}
else
  echo "Now you can push the image with: docker push ${IMAGE}"
fi
