#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

IMAGE=apolloauto/internal:apollo-cluster_$(date +%Y%m%d-%H%M)
docker build --network host -t ${IMAGE} .
