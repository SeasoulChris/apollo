#!/usr/bin/env bash

# Input.
LOCAL_JOB_FILE="$1"

# Config.
IMAGE="apolloauto/spark:20190209_1155"
K8S="https://180.76.185.100:6443"
WORKERS=1
CORES=1
CONDA_ENV="py27"
AWS_KEY="<INPUT>"
AWS_SEC="<INPUT>"

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
    --conf spark.kubernetes.authenticate.driver.serviceAccountName="spark" \
    --conf spark.kubernetes.container.image="${IMAGE}" \
    --conf spark.kubernetes.container.image.pullSecrets="dockerhub.com" \
    --conf spark.kubernetes.executor.request.cores="${CORES}" \
    --conf spark.kubernetes.pyspark.pythonVersion=2 \
    --conf spark.executorEnv.AWS_ACCESS_KEY_ID="${AWS_KEY}" \
    --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY="${AWS_SEC}" \
    --conf spark.executorEnv.APOLLO_BOS_BUCKET="apollo-platform" \
    --conf spark.executorEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.kubernetes.driverEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --py-files "${REMOTE_FUELING_PKG}" \
    "${REMOTE_JOB_FILE}"
