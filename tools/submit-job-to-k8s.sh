#!/usr/bin/env bash

# Current cluster resources (Show usage with "kubectl top nodes"):
#   CPU Cores: 64
#   Memory: 500GB
#   Ephemeral Storage: 2TB

# Default value for configurable arguments.
JOB_FILE=""
IMAGE="hub.baidubce.com/apollo/spark:latest"
CONDA_ENV="fuel-py27-cyber"
EXECUTORS=20
EXECUTOR_CORES=2
EXECUTOR_MEMORY=20g
MEMORY_OVERHEAD_FACTOR=0
# NON_JVM_MEMORY = EXECUTOR_MEMORY * MEMORY_OVERHEAD_FACTOR
# Check https://spark.apache.org/docs/latest/running-on-kubernetes.html for more
# information.

while [ $# -gt 0 ]; do
  case "$1" in
    --image|-i)
      shift
      IMAGE=$1
      ;;
    --env|-e)
      shift
      CONDA_ENV=$1
      ;;
    --workers|-w)
      shift
      EXECUTORS=$1
      ;;
    --cpu|-c)
      shift
      EXECUTOR_CORES=$1
      ;;
    --memory|-m)
      shift
      EXECUTOR_MEMORY=$1
      ;;
    --memory-overhead)
      shift
      MEMORY_OVERHEAD_FACTOR=$1
      ;;
    *)
      if [ -f "$1" ]; then
        JOB_FILE=$1
      else
        echo -e "$1: Unknown option or file not exists."
        exit 1
      fi
      ;;
  esac
  shift
done

if [ -z "${JOB_FILE}" ]; then
  echo "No job specified."
  exit 1
fi

# Generally fixed config.
K8S="https://180.76.98.43:6443"
DRIVER_MEMORY=2g
APOLLO_SPARK_REPO="$(cd $( dirname "${BASH_SOURCE[0]}" )/../../apollo-spark; pwd)"
BOS_FSTOOL_EXECUTABLE="$( dirname "${BASH_SOURCE[0]}" )/../apps/static/bos_fstool"
# End of config.

set -x
set -e

# Upload local files to remote.
BOS_MOUNT_PATH="/mnt/bos"
REMOTE_JOB_PATH="modules/data/jobs/$(date +%Y%m%d-%H%M)_${USER}"
REMOTE_JOB_FILE="${REMOTE_JOB_PATH}/$(basename ${JOB_FILE})"
REMOTE_FUELING_PKG="${REMOTE_JOB_PATH}/fueling.zip"

"${BOS_FSTOOL_EXECUTABLE}" -s "${JOB_FILE}" -d "${REMOTE_JOB_FILE}"
REMOTE_JOB_FILE="${BOS_MOUNT_PATH}/${REMOTE_JOB_FILE}"

EVENTS_LOG_PATH=${BOS_MOUNT_PATH}/modules/data/spark/spark-events

pushd "$( dirname "${BASH_SOURCE[0]}" )/.."
  LOCAL_FUELING_PKG=".fueling.zip"
  rm -f "${LOCAL_FUELING_PKG}"
  zip -r "${LOCAL_FUELING_PKG}" ./fueling -x *.pyc */__pycache__
  "${BOS_FSTOOL_EXECUTABLE}" -s "${LOCAL_FUELING_PKG}" -d "${REMOTE_FUELING_PKG}" 
  REMOTE_FUELING_PKG="${BOS_MOUNT_PATH}/${REMOTE_FUELING_PKG}"
popd

# Submit job with fueling package.
"${APOLLO_SPARK_REPO}/bin/spark-submit" \
    --master "k8s://${K8S}" \
    --deploy-mode cluster \
    --conf spark.default.parallelism="${EXECUTORS}" \
    --conf spark.driver.memory="${DRIVER_MEMORY}" \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=file:${EVENTS_LOG_PATH} \
    --conf spark.executor.instances="${EXECUTORS}" \
    --conf spark.executor.memory="${EXECUTOR_MEMORY}" \
    --conf spark.kubernetes.memoryOverheadFactor="${MEMORY_OVERHEAD_FACTOR}" \
\
    --conf spark.kubernetes.authenticate.driver.serviceAccountName="spark" \
    --conf spark.kubernetes.container.image="${IMAGE}" \
    --conf spark.kubernetes.container.image.pullPolicy="Always" \
    --conf spark.kubernetes.container.image.pullSecrets="baidubce" \
    --conf spark.kubernetes.executor.request.cores="${EXECUTOR_CORES}" \
\
    --conf spark.executorEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.executorEnv.APOLLO_FUELING_PYPATH="${REMOTE_FUELING_PKG}" \
    --conf spark.kubernetes.driverEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.kubernetes.driverEnv.APOLLO_EXECUTORS="${EXECUTORS}" \
    --conf spark.kubernetes.driverEnv.APOLLO_FUELING_PYPATH="${REMOTE_FUELING_PKG}" \
    --conf spark.kubernetes.driver.secretKeyRef.APOLLO_EMAIL_PASSWD="apollo-k8s-secret:email-passwd" \
    --conf spark.kubernetes.driver.secretKeyRef.AWS_ACCESS_KEY_ID="bos-secret:ak" \
    --conf spark.kubernetes.driver.secretKeyRef.AWS_SECRET_ACCESS_KEY="bos-secret:sk" \
    --conf spark.kubernetes.driver.secretKeyRef.MONGO_USER="mongo-secret:mongo-user" \
    --conf spark.kubernetes.driver.secretKeyRef.MONGO_PASSWD="mongo-secret:mongo-passwd" \
    --conf spark.kubernetes.executor.secretKeyRef.AWS_ACCESS_KEY_ID="bos-secret:ak" \
    --conf spark.kubernetes.executor.secretKeyRef.AWS_SECRET_ACCESS_KEY="bos-secret:sk" \
    --conf spark.kubernetes.executor.secretKeyRef.MONGO_USER="mongo-secret:mongo-user" \
    --conf spark.kubernetes.executor.secretKeyRef.MONGO_PASSWD="mongo-secret:mongo-passwd" \
\
    "${REMOTE_JOB_FILE}"
