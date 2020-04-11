#!/usr/bin/env bash

DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollofuel"

REPO="${DOCKER_REGISTRY}/${DOCKER_USER}/afs-data-service"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"

cd "$( dirname "${BASH_SOURCE[0]}" )"
set -ex

# Prepare source
SRC_DIR="server"
mkdir -p ${SRC_DIR}
cp ../server.py ${SRC_DIR}
cp ../proto/* ${SRC_DIR}

# Build.
docker build -t ${IMAGE} --network host .
rm -rf ${SRC_DIR}
docker push ${IMAGE}

# Deploy.
cp ~/.kube/config kube.config.original
sudo cp kube.config ~/.kube/config
sed -i "s|image: ${REPO}.*|image: ${IMAGE}|g" deploy.yaml
kubectl apply -f deploy.yaml --validate=false
sudo mv kube.config.original ~/.kube/config

