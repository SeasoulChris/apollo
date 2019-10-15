#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

LOCAL_FUELING_PKG="deploy/fueling-latest.zip"
REMOTE_FUELING_PKG="/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip"
rm -f "${LOCAL_FUELING_PKG}"

set -e

zip -r "${LOCAL_FUELING_PKG}" ./fueling -x *.pyc */__pycache__
echo "Now you can push the package ${LOCAL_FUELING_PKG} to ${REMOTE_FUELING_PKG}"
