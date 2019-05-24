#!/usr/bin/env bash

DOCKER_REGISTRY="registry.bce.baidu.com"
REPO="${DOCKER_REGISTRY}/a7e094e2914d424caa523a201e57995b/apollo-fuel-bae-proxy"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"
# Go to apollo-fuel root.
cd $( dirname "${BASH_SOURCE[0]}" )/../../../..

cp ~/.kube/config ./kube.config
# Generate HTTPS keys. The common name should be the host domain.
# openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 3650
#    -subj "/C=CN/ST=Beijing/L=Beijing/O=Baidu/OU=IDG/CN=apollo.bceapp.com"
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
        docker run -it --rm --net host ${IMAGE} --debug"
  echo "Or push the image with:
        docker push ${IMAGE}"
fi
