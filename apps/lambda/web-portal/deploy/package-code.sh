#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )/.."

TIME=$(date +%Y%m%d-%H%M)
PACKAGE="deploy/web-portal_${TIME}.zip"

zip -r "${PACKAGE}" *.py *.sh templates/ ssl_keys/
echo "Packaged ${PACKAGE}"
