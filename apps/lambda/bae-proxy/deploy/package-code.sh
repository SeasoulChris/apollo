#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )/.."

TIME=$(date +%Y%m%d-%H%M)
PACKAGE="deploy/bae-proxy_${TIME}.zip"

zip -r "${PACKAGE}" *.py *.sh ssl_keys/
echo "Packaged ${PACKAGE}"
