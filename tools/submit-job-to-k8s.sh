#!/usr/bin/env bash

# Input.
LOCAL_JOB_FILE="$1"

# Config.
# TODO(xiangquan): We got problem on pulling from private repo. Use a public
# repo for now as there is no confidential things in the image.
IMAGE=xiangquan/spark:20190207_1717
K8S=https://180.76.185.100:6443
WORKERS=2

set -x
set -e

# Upload local files to remote.
REMOTE_JOB_PATH="/mnt/bos/modules/data/jobs"
REMOTE_PREFIX="${REMOTE_JOB_PATH}/$(date +%Y%m%d-%H%M)_${USER}"
REMOTE_JOB_FILE="${REMOTE_PREFIX}_$(basename ${LOCAL_JOB_FILE})"
REMOTE_FUELING_PKG="${REMOTE_PREFIX}_fueling.zip"

sudo mkdir -p "${REMOTE_JOB_PATH}"
sudo cp "${LOCAL_JOB_FILE}" "${REMOTE_JOB_FILE}"

pushd "$( dirname "${BASH_SOURCE[0]}" )/.."
  LOCAL_FUELING_PKG=".fueling.zip"
  rm -f "${LOCAL_FUELING_PKG}" && \
  zip -r "${LOCAL_FUELING_PKG}" fueling -x *.pyc && \
  sudo cp "${LOCAL_FUELING_PKG}" "${REMOTE_FUELING_PKG}"
popd

# Submit job with fueling package.
spark-submit \
    --master "k8s://${K8S}" \
    --deploy-mode cluster \
    --conf spark.executor.instances="${WORKERS}" \
    --conf spark.kubernetes.container.image="${IMAGE}" \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName="spark" \
    --conf spark.kubernetes.container.image.pullSecrets="dockerhub.com" \
    --py-files "${REMOTE_FUELING_PKG}" \
    "${REMOTE_JOB_FILE}"
