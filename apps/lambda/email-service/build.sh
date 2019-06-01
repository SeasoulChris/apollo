#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

TIME=$(date +%Y%m%d-%H%M)

mkdir -p deploy
find . -type f | grep -v ./deploy/ | xargs zip deploy/email-service_${TIME}.zip
