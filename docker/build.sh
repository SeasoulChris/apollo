#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE=apolloauto/apollo:k8s-spark

echo "Building ${IMAGE}..."
docker build --network=host -t ${IMAGE} .
