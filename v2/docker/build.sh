#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

IMAGE=local:fuel
docker build --network=host -t ${IMAGE} .
