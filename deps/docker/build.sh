#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"/..

set -e

IMAGE=apolloauto/fuel-client:$(date +%Y%m%d_%H%M)
docker build --network=host -t ${IMAGE} -f docker/Dockerfile .
docker run --rm ${IMAGE} /usr/local/miniconda/bin/conda list -n fuel | \
    python3 generate_conda_env_lock.py docker/fuel.yaml docker/fuel.yaml.lock

sed -i "s|IMAGE=.*|IMAGE=\"${IMAGE}\"|g" ../tools/docker_version.sh
