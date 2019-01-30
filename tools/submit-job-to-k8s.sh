#!/usr/bin/env bash

LOCAL_JOB="$1"

# Config.
IMAGE=apolloauto/spark:20190129_0000
K8S=https://180.76.185.100:6443
WORKERS=5

set -x
set -e

# Upload job files to BOS.
JOB_PATH="/mnt/bos/modules/data/jobs"
PREFIX="${JOB_PATH}/$(date +%Y%m%d-%H%M)_${USER}"
JOB_FILE="${PREFIX}_$(basename ${LOCAL_JOB})"
FUELING_PKG="${PREFIX}_fueling.zip"

sudo mkdir -p "${JOB_PATH}"
sudo cp "${LOCAL_JOB}" "${JOB_FILE}"

pushd "$( dirname "${BASH_SOURCE[0]}" )/.."
  rm -f fueling.zip && \
  zip -r fueling.zip fueling && \
  sudo cp fueling.zip "${FUELING_PKG}"
popd

# Submit job with fueling package.
spark-submit \
    --master k8s://${K8S} \
    --deploy-mode cluster \
    --conf spark.executor.instances=${WORKERS} \
    --conf spark.kubernetes.container.image=${IMAGE} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --py-files "${FUELING_PKG}" \
    ${JOB_FILE}
