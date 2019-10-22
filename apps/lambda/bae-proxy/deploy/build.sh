#!/usr/bin/env bash

DOCKER_REGISTRY="registry.bce.baidu.com"
REPO="${DOCKER_REGISTRY}/a7e094e2914d424caa523a201e57995b/apollo-fuel-bae-proxy"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"
# Go to apollo-fuel root.
cd $( dirname "${BASH_SOURCE[0]}" )/../../../..

set -e

# Build your local apollo.
bash ../apollo/apollo_docker.sh build_py
cp -r ../apollo/py_proto ./
cp ~/.kube/config ./kube.config

docker build -t ${IMAGE} --network host -f apps/lambda/bae-proxy/deploy/Dockerfile .

# Please provide credential if you want to login automatically.
DOCKER_USER=""
DOCKER_PASSWORD=""
if [ ! -z "${DOCKER_PASSWORD}" ]; then
  docker login -u ${DOCKER_USER} -p ${DOCKER_PASSWORD} ${DOCKER_REGISTRY}
fi

if [ "$1" = "push" ]; then
  docker push ${IMAGE}
else
  echo "Now you can test the image with:
        docker run -it --rm --net host -v $(pwd)/apps/lambda/bae-proxy:/home/bae/app ${IMAGE} --debug"
fi
