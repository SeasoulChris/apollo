#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

TIME=$(date +%Y%m%d-%H%M)

mkdir -p generated
find . -type f | grep -v ./generated/ | xargs zip generated/bae-proxy_${TIME}.zip
