#!/usr/bin/env bash

# Input.
LOCAL_JOB_FILE="$1"

# Config.
IMAGE="xiangquan/spark:20190222_1046"
K8S="https://180.76.98.43:6443"
DRIVER_MEMORY=2g
EXECUTORS=10
EXECUTOR_CORES=2
EXECUTOR_MEMORY=16g
CONDA_ENV="py27"
AWS_KEY="<INPUT>"
AWS_SEC="<INPUT>"
APOLLO_SPARK_REPO="$(cd $( dirname "${BASH_SOURCE[0]}" )/../../apollo-spark; pwd)"
APOLLO_ENABLED="yes"
# End of config.

set -x
set -e

# Upload local files to remote.
REMOTE_JOB_PATH="/mnt/bos/modules/data/jobs/$(date +%Y%m%d-%H%M)_${USER}"
REMOTE_JOB_FILE="${REMOTE_JOB_PATH}/$(basename ${LOCAL_JOB_FILE})"
REMOTE_FUELING_PKG="${REMOTE_JOB_PATH}/fueling.zip"

sudo mkdir -p "${REMOTE_JOB_PATH}"
sudo cp "${LOCAL_JOB_FILE}" "${REMOTE_JOB_FILE}"

pushd "$( dirname "${BASH_SOURCE[0]}" )/.."
  LOCAL_FUELING_PKG=".fueling.zip"
  rm -f "${LOCAL_FUELING_PKG}" && \
  zip -r "${LOCAL_FUELING_PKG}" ./fueling -x *.pyc && \
  sudo cp "${LOCAL_FUELING_PKG}" "${REMOTE_FUELING_PKG}"
popd

# Submit job with fueling package.
sudo "${APOLLO_SPARK_REPO}/bin/spark-submit" \
    --master "k8s://${K8S}" \
    --deploy-mode cluster \
    --conf spark.driver.memory="${DRIVER_MEMORY}" \
    --conf spark.executor.instances="${EXECUTORS}" \
    --conf spark.executor.memory="${EXECUTOR_MEMORY}" \
\
    --conf spark.kubernetes.authenticate.driver.serviceAccountName="spark" \
    --conf spark.kubernetes.container.image="${IMAGE}" \
    --conf spark.kubernetes.executor.request.cores="${EXECUTOR_CORES}" \
\
    --conf spark.executorEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.executorEnv.APOLLO_ENABLED="${APOLLO_ENABLED}" \
    --conf spark.executorEnv.APOLLO_FUELING_PYPATH="${REMOTE_FUELING_PKG}" \
    --conf spark.executorEnv.AWS_ACCESS_KEY_ID="${AWS_KEY}" \
    --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY="${AWS_SEC}" \
    --conf spark.kubernetes.driverEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.kubernetes.driverEnv.APOLLO_ENABLED="${APOLLO_ENABLED}" \
    --conf spark.kubernetes.driverEnv.APOLLO_EXECUTORS="${EXECUTORS}" \
    --conf spark.kubernetes.driverEnv.APOLLO_FUELING_PYPATH="${REMOTE_FUELING_PKG}" \
    --conf spark.kubernetes.driverEnv.AWS_ACCESS_KEY_ID="${AWS_KEY}" \
    --conf spark.kubernetes.driverEnv.AWS_SECRET_ACCESS_KEY="${AWS_SEC}" \
\
    "${REMOTE_JOB_FILE}"
