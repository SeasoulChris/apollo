#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

IMAGE="local:image"
docker build --network=host -t ${IMAGE} .

# Re-tag it per your need and then push.
