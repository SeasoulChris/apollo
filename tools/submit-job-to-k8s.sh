#!/usr/bin/env bash

# Current cluster resources (Show usage with "kubectl top nodes"):
#   CPU Cores: 64
#   Memory: 500GB
#   Ephemeral Storage: 2TB

# Default config.
JOB_FILE=""
CONDA_ENV="fuel-py27-cyber"
EXECUTORS=16
EXECUTOR_CORES=3
EXECUTOR_MEMORY=24g
IMAGE="xiangquan/spark:20190311_1741"

while [ $# -gt 0 ]; do
    case "$1" in
    --env)
        shift
        CONDA_ENV=$1
        ;;
    --job)
        shift
        JOB_FILE=$1
        ;;
    --workers)
        shift
        EXECUTORS=$1
        ;;
    --worker-cpu)
        shift
        EXECUTOR_CORES=$1
        ;;
    --worker-memory)
        shift
        EXECUTOR_MEMORY=$1
        ;;
    --image)
        shift
        IMAGE=$1
        ;;
    *)
        echo -e "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

# Config.
K8S="https://180.76.98.43:6443"
DRIVER_MEMORY=2g
AWS_KEY="<INPUT>"
AWS_SEC="<INPUT>"
APOLLO_SPARK_REPO="$(cd $( dirname "${BASH_SOURCE[0]}" )/../../apollo-spark; pwd)"
# End of config.

set -x
set -e

if [ -z "${JOB_FILE}" ]; then
  echo "No --job specified."
  exit 1
fi

# Upload local files to remote.
REMOTE_JOB_PATH="/mnt/bos/modules/data/jobs/$(date +%Y%m%d-%H%M)_${USER}"
REMOTE_JOB_FILE="${REMOTE_JOB_PATH}/$(basename ${JOB_FILE})"
REMOTE_FUELING_PKG="${REMOTE_JOB_PATH}/fueling.zip"

sudo mkdir -p "${REMOTE_JOB_PATH}"
sudo cp "${JOB_FILE}" "${REMOTE_JOB_FILE}"

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
    --conf spark.default.parallelism="${EXECUTORS}" \
    --conf spark.driver.memory="${DRIVER_MEMORY}" \
    --conf spark.executor.instances="${EXECUTORS}" \
    --conf spark.executor.memory="${EXECUTOR_MEMORY}" \
\
    --conf spark.kubernetes.authenticate.driver.serviceAccountName="spark" \
    --conf spark.kubernetes.container.image="${IMAGE}" \
    --conf spark.kubernetes.executor.request.cores="${EXECUTOR_CORES}" \
\
    --conf spark.executorEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.executorEnv.APOLLO_FUELING_PYPATH="${REMOTE_FUELING_PKG}" \
    --conf spark.executorEnv.AWS_ACCESS_KEY_ID="${AWS_KEY}" \
    --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY="${AWS_SEC}" \
    --conf spark.kubernetes.driverEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.kubernetes.driverEnv.APOLLO_EXECUTORS="${EXECUTORS}" \
    --conf spark.kubernetes.driverEnv.APOLLO_FUELING_PYPATH="${REMOTE_FUELING_PKG}" \
    --conf spark.kubernetes.driverEnv.AWS_ACCESS_KEY_ID="${AWS_KEY}" \
    --conf spark.kubernetes.driverEnv.AWS_SECRET_ACCESS_KEY="${AWS_SEC}" \
\
    "${REMOTE_JOB_FILE}"
