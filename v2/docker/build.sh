#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

IMAGE=apolloauto/fuel-client:$(date +%Y%m%d_%H%M)
docker build --network=host -t ${IMAGE} .
