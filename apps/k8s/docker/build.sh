#!/usr/bin/env bash

REPO="hub.baidubce.com/apollo/warehouse"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"
IMAGE_ALIAS="${REPO}:latest"

echo "Building image: ${IMAGE}"
# Go to apollo-fuel root.
cd $( dirname "${BASH_SOURCE[0]}" )/../../..

set -e
set -x

# Build your local apollo.
bash ../apollo/apollo_docker.sh build_py
cp -r ../apollo/py_proto apps/k8s/docker/

docker build -t ${IMAGE} --network host -f apps/k8s/docker/Dockerfile .
docker tag ${IMAGE} ${IMAGE_ALIAS}

# Login.
DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollo"
DOCKER_PASSWORD=""
docker login -u ${DOCKER_USER} -p ${DOCKER_PASSWORD} ${DOCKER_REGISTRY}

if [ "$1" = "push" ]; then
  docker push ${IMAGE}
  docker push ${IMAGE_ALIAS}
else
  echo "Now you can push the images with:
      docker push ${IMAGE}
      docker push ${IMAGE_ALIAS}
  "
fi
