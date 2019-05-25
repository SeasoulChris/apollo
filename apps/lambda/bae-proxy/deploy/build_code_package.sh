#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )/.."

TIME=$(date +%Y%m%d-%H%M)

zip -r deploy/bae-proxy_${TIME}.zip *.py ssl_keys/
