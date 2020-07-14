#!/usr/bin/env bash

IMAGE="kube_proxy:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"

cd $( dirname "${BASH_SOURCE[0]}" )
set -e

cp ~/.kube/config .
docker build -t ${IMAGE} --network host .
# Here we use 8081 as the host service port. If port conflict occurs, change it to any other available one.
docker run -dit -p 8081:30001 ${IMAGE}
