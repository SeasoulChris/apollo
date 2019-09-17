#!/usr/bin/env bash

# Guideline for resource allocation:
# 1. Maximum resource of an EXECUTOR is the resource of a node, i.e.:
#    CPU Cores: 32
#    Memory: 256GB
#    Ephemeral Storage: 5TB
# 2. Maximum resource of a JOB is the total resource of all 5 nodes, i.e.:
#    CPU Cores: 160
#    Memory: 1.25TB
#    Ephemeral Storage: 25TB
# Show current cluster resource usage with "kubectl top nodes".

# Quick check.
if [ ! -f ${HOME}/.kube/config ]; then
  echo "You must run on host and have valid k8s config."
  exit 1
fi
if [ ! -d "./fueling" ]; then
  echo "You must run from apollo-fuel root folder."
  exit 1
fi
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "${BASH_SOURCE[0]} [options] <job.py> [job-gflags]"
  echo ""
  echo "Options are:"
  grep "^    --" "${BASH_SOURCE[0]}" | grep ')' | sed 's/)//' | sed 's/|/,/'
  exit 0
fi

if [ "${IN_CLIENT_DOCKER}" != "true" ]; then
  docker run --rm --net host \
      -v "${HOME}/.kube":/root/.kube \
      -v "$(pwd)":/fuel \
      -e IN_CLIENT_DOCKER=true \
      -e SUBMITTER=$(whoami) \
      -w="/fuel" \
      apolloauto/fuel-client:20190821_1718 \
      bash tools/submit-job-to-k8s.sh $@
  exit $?
fi

# Now we are inside the client docker, and working dir is the fuel root.
set -e

# Default value for configurable arguments.
JOB_FILE=""
FUELING_PKG=""
IMAGE="hub.baidubce.com/apollo/spark:latest"
CONDA_ENV="fuel-py27-cyber"
EXECUTORS=1
EXECUTOR_CORES=1
EXECUTOR_MEMORY=12g
EXECUTOR_DISK_GB=20
COMPUTE_TYPE="CPU"
LOG_VERBOSITY="INFO"
# Configurable but rarely change.
MEMORY_OVERHEAD_FACTOR=0
BOS_REGION="bj"
BOS_BUCKET="apollo-platform"

# Partner BOS config.
PARTNER_BOS_REGION=""
PARTNER_BOS_BUCKET=""
PARTNER_BOS_ACCESS=""
PARTNER_BOS_SECRET=""

# Partner Azure config.
AZURE_STORAGE_ACCOUNT=""
AZURE_STORAGE_ACCESS_KEY=""
AZURE_BLOB_CONTAINER=""

# NON_JVM_MEMORY = EXECUTOR_MEMORY * MEMORY_OVERHEAD_FACTOR
# Check https://spark.apache.org/docs/latest/running-on-kubernetes.html for more
# information.

while [ $# -gt 0 ]; do
  case "$1" in
    --image|-i)                   # Docker image: "-i hub.baidubce.com/apollo/spark:latest"
      shift
      IMAGE=$1
      ;;
    --env|-e)                     # Conda environment: "-e fuel-py36"
      shift
      CONDA_ENV=$1
      ;;
    --workers|-w)                 # Worker count: "-w 8"
      shift
      EXECUTORS=$1
      ;;
    --cpu|-c)                     # CPU count per worker: "-c 2"
      shift
      EXECUTOR_CORES=$1
      ;;
    --gpu|-g)                     # Whether to use GPU worker.
      COMPUTE_TYPE="GPU"
      ;;
    --memory|-m)                  # Memory per worker: "-m 24g"
      shift
      EXECUTOR_MEMORY=$1
      ;;
    --memory-overhead)            # Non JVM memory per worker: "--memory-overhead 0"
      shift
      MEMORY_OVERHEAD_FACTOR=$1
      ;;
    --disk|-d)                    # Disk size in GB per worker: "-d 50"
      shift
      EXECUTOR_DISK_GB=$1
      ;;
    --verbosity|-v)               # Log verbosity: "-v INFO", or DEBUG, WARNING, ERROR, FATAL.
      shift
      LOG_VERBOSITY=$1
      ;;
    --fueling)                    # Pre packaged fueling folder to avoid uploading.
      shift
      FUELING_PKG=$1
      ;;
    --bos_region)                 # Our BOS region.
      shift
      BOS_REGION=$1
      ;;
    --bos_bucket)                 # Our BOS bucket.
      shift
      BOS_BUCKET=$1
      ;;
    --partner_bos_region)         # Partner BOS region.
      shift
      PARTNER_BOS_REGION=$1
      ;;
    --partner_bos_bucket)         # Partner BOS bucket.
      shift
      PARTNER_BOS_BUCKET=$1
      ;;
    --partner_bos_access)         # Partner BOS access key.
      shift
      PARTNER_BOS_ACCESS=$1
      ;;
    --partner_bos_secret)         # Partner BOS secret key.
      shift
      PARTNER_BOS_SECRET=$1
      ;;
    --azure_storage_account)      # Azure storage account.
      shift
      AZURE_STORAGE_ACCOUNT=$1
      ;;
    --azure_storage_access_key)   # Azure storage access key.
      shift
      AZURE_STORAGE_ACCESS_KEY=$1
      ;;
    --azure_blob_container)       # Partner BOS access key.
      shift
      AZURE_BLOB_CONTAINER=$1
      ;;
    *)
      JOB_FILE=$1
      echo "Spawning job: ${JOB_FILE}"
      shift
      break
      ;;
  esac
  shift
done

if [ -z "${JOB_FILE}" ]; then
  echo "No job specified."
  exit 1
fi

# Generally fixed config.
K8S=$(grep 'server:' ~/.kube/config | awk '{print $2}')
DRIVER_MEMORY=2g
BOS_FSTOOL_EXECUTABLE="./apps/local/bos_fstool"
BOS_MOUNT_PATH="/mnt/bos"
EVENTS_LOG_PATH=${BOS_MOUNT_PATH}/modules/data/spark/spark-events
# End of config.

REMOTE_JOB_PATH="modules/data/jobs/$(date +%Y%m%d-%H%M%S)_${SUBMITTER}"
if [ -f ${JOB_FILE} ]; then
  # Upload local job file to remote.
  REMOTE_JOB_FILE="${REMOTE_JOB_PATH}/$(basename ${JOB_FILE})"
  "${BOS_FSTOOL_EXECUTABLE}" -s "${JOB_FILE}" -d "${REMOTE_JOB_FILE}"
  JOB_FILE="${BOS_MOUNT_PATH}/${REMOTE_JOB_FILE}"
fi

if [ -z ${FUELING_PKG} ]; then
  # Upload local fueling package to remote.
  REMOTE_FUELING_PKG="${REMOTE_JOB_PATH}/fueling.zip"
  LOCAL_FUELING_PKG="/tmp/fueling.zip"
  rm -f "${LOCAL_FUELING_PKG}"
  zip -r "${LOCAL_FUELING_PKG}" ./fueling -x *.pyc */__pycache__
  "${BOS_FSTOOL_EXECUTABLE}" -s "${LOCAL_FUELING_PKG}" -d "${REMOTE_FUELING_PKG}" 
  FUELING_PKG="${BOS_MOUNT_PATH}/${REMOTE_FUELING_PKG}"
fi

ENV_CONF=""
ENVS=(
  "APOLLO_CONDA_ENV=${CONDA_ENV}"
  "APOLLO_FUELING_PYPATH=${FUELING_PKG}"
  "APOLLO_COMPUTE_TYPE=${COMPUTE_TYPE}"
  "APOLLO_EXECUTORS=${EXECUTORS}"
  "LOG_VERBOSITY=${LOG_VERBOSITY}"
  # Partner BOS support.
  "PARTNER_BOS_REGION=${PARTNER_BOS_REGION}"
  "PARTNER_BOS_BUCKET=${PARTNER_BOS_BUCKET}"
  "PARTNER_BOS_ACCESS=${PARTNER_BOS_ACCESS}"
  "PARTNER_BOS_SECRET=${PARTNER_BOS_SECRET}"
  # Azure Blob support.
  "AZURE_STORAGE_ACCOUNT=${AZURE_STORAGE_ACCOUNT}"
  "AZURE_STORAGE_ACCESS_KEY=${AZURE_STORAGE_ACCESS_KEY}"
  "AZURE_BLOB_CONTAINER=${AZURE_BLOB_CONTAINER}"
)
for i in ${ENVS[@]}; do
  IFS='=' read KEY VALUE <<< "${i}"
  if [ ! -z ${VALUE} ]; then
    ENV_CONF="${ENV_CONF} \
        --conf spark.executorEnv.${KEY}=${VALUE} \
        --conf spark.kubernetes.driverEnv.${KEY}=${VALUE}"
  fi
done

SECRET_ENVS=(
  "BOS_ACCESS=bos-secret:ak"
  "BOS_SECRET=bos-secret:sk"
  "BOS_BUCKET=bos-secret:bucket"
  "BOS_REGION=bos-secret:region"
  "MONGO_USER=mongo-secret:user"
  "MONGO_PASSWD=mongo-secret:passwd"
  "OUTLOOK_USER=outlook-secret:user"
  "OUTLOOK_PASSWD=outlook-secret:passwd"
)
for i in ${SECRET_ENVS[@]}; do
  IFS='=' read KEY VALUE <<< "${i}"
  ENV_CONF="${ENV_CONF} \
      --conf spark.kubernetes.driver.secretKeyRef.${KEY}=${VALUE} \
      --conf spark.kubernetes.executor.secretKeyRef.${KEY}=${VALUE}"
done

JOB_NAME=${SUBMITTER}-$(basename "${JOB_FILE}" | cut -d "." -f 1 | sed "s/_/-/g")

./tools/k8s-job-watcher.sh "${JOB_NAME}" &

# Submit job with fueling package.
spark-submit \
    --master "k8s://${K8S}" \
    --name ${JOB_NAME} \
    --deploy-mode cluster \
    --conf spark.default.parallelism="${EXECUTORS}" \
    --conf spark.driver.memory="${DRIVER_MEMORY}" \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=file:${EVENTS_LOG_PATH} \
    --conf spark.executor.instances="${EXECUTORS}" \
    --conf spark.executor.memory="${EXECUTOR_MEMORY}" \
    --conf spark.kubernetes.memoryOverheadFactor="${MEMORY_OVERHEAD_FACTOR}" \
    --conf spark.kubernetes.node.selector.computetype="${COMPUTE_TYPE}" \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName="spark" \
    --conf spark.kubernetes.container.image="${IMAGE}" \
    --conf spark.kubernetes.container.image.pullPolicy="Always" \
    --conf spark.kubernetes.container.image.pullSecrets="baidubce" \
    --conf spark.kubernetes.executor.ephemeralStorageGB="${EXECUTOR_DISK_GB}" \
    --conf spark.kubernetes.executor.request.cores="${EXECUTOR_CORES}" \
    ${ENV_CONF} "${JOB_FILE}" --running_mode=PROD $@ | tee /tmp/spark-submit.log

EXIT_CODE=$(grep 'Exit code: ' /tmp/spark-submit.log | awk '{print $3}')
exit ${EXIT_CODE}
