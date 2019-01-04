#!/usr/bin/env bash

# Fail on first failure.
set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Change the Apollo directory if it's different in your environment.
APOLLO_DIR="$( cd "${DIR}/../../apollo" && pwd )"


bash ${APOLLO_DIR}/apollo_docker.sh build_py
cp -r "${APOLLO_DIR}/py_proto/modules" "${DIR}/"
cp -r "${APOLLO_DIR}/py_proto/cyber" "${DIR}/"
cp -r "${APOLLO_DIR}/cyber/python/cyber_py" "${DIR}/"
