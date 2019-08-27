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
      -e SUBMITTER=${USER} \
      -w="/fuel" \
      apolloauto/fuel-client:20190821_1718 \
      bash tools/submit-job-to-k8s.sh $@
  exit $?
fi

# Now we are inside the client docker.
set -e
cd /fuel

# Default value for configurable arguments.
JOB_FILE=""
FUELING_PKG=""
IMAGE="hub.baidubce.com/apollo/spark:latest"
CONDA_ENV="fuel-py27-cyber"
EXECUTORS=8
EXECUTOR_CORES=2
EXECUTOR_MEMORY=24g
EXECUTOR_DISK_GB=50
MEMORY_OVERHEAD_FACTOR=0

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

# Compute type, default is CPU if GPU is not explicitly specified
COMPUTE_TYPE="CPU"


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
    --fueling)                    # Pre packaged fueling folder to avoid uploading.
      shift
      FUELING_PKG=$1
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
K8S="https://180.76.98.43:6443"
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

# Submit job with fueling package.
spark-submit \
    --master "k8s://${K8S}" \
    --deploy-mode cluster \
    --conf spark.default.parallelism="${EXECUTORS}" \
    --conf spark.driver.memory="${DRIVER_MEMORY}" \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=file:${EVENTS_LOG_PATH} \
    --conf spark.executor.instances="${EXECUTORS}" \
    --conf spark.executor.memory="${EXECUTOR_MEMORY}" \
    --conf spark.kubernetes.memoryOverheadFactor="${MEMORY_OVERHEAD_FACTOR}" \
    --conf spark.kubernetes.node.selector.computetype="${COMPUTE_TYPE}" \
\
    --conf spark.kubernetes.authenticate.driver.serviceAccountName="spark" \
    --conf spark.kubernetes.container.image="${IMAGE}" \
    --conf spark.kubernetes.container.image.pullPolicy="Always" \
    --conf spark.kubernetes.container.image.pullSecrets="baidubce" \
    --conf spark.kubernetes.executor.ephemeralStorageGB="${EXECUTOR_DISK_GB}" \
    --conf spark.kubernetes.executor.request.cores="${EXECUTOR_CORES}" \
\
    --conf spark.executorEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.executorEnv.APOLLO_FUELING_PYPATH="${FUELING_PKG}" \
    --conf spark.executorEnv.APOLLO_COMPUTE_TYPE="${COMPUTE_TYPE}" \
    --conf spark.kubernetes.driverEnv.APOLLO_CONDA_ENV="${CONDA_ENV}" \
    --conf spark.kubernetes.driverEnv.APOLLO_EXECUTORS="${EXECUTORS}" \
    --conf spark.kubernetes.driverEnv.APOLLO_FUELING_PYPATH="${FUELING_PKG}" \
    --conf spark.kubernetes.driverEnv.APOLLO_COMPUTE_TYPE="${COMPUTE_TYPE}" \
\
    --conf spark.kubernetes.driver.secretKeyRef.AWS_ACCESS_KEY_ID="bos-secret:ak" \
    --conf spark.kubernetes.driver.secretKeyRef.AWS_SECRET_ACCESS_KEY="bos-secret:sk" \
    --conf spark.kubernetes.executor.secretKeyRef.AWS_ACCESS_KEY_ID="bos-secret:ak" \
    --conf spark.kubernetes.executor.secretKeyRef.AWS_SECRET_ACCESS_KEY="bos-secret:sk" \
\
    --conf spark.kubernetes.driver.secretKeyRef.MONGO_USER="mongo-secret:mongo-user" \
    --conf spark.kubernetes.driver.secretKeyRef.MONGO_PASSWD="mongo-secret:mongo-passwd" \
    --conf spark.kubernetes.executor.secretKeyRef.MONGO_USER="mongo-secret:mongo-user" \
    --conf spark.kubernetes.executor.secretKeyRef.MONGO_PASSWD="mongo-secret:mongo-passwd" \
\
    --conf spark.kubernetes.driverEnv.AZURE_STORAGE_ACCOUNT=${AZURE_STORAGE_ACCOUNT} \
    --conf spark.kubernetes.driverEnv.AZURE_STORAGE_ACCESS_KEY=${AZURE_STORAGE_ACCESS_KEY} \
    --conf spark.kubernetes.driverEnv.AZURE_BLOB_CONTAINER=${AZURE_BLOB_CONTAINER} \
    --conf spark.executorEnv.AZURE_STORAGE_ACCOUNT=${AZURE_STORAGE_ACCOUNT} \
    --conf spark.executorEnv.AZURE_STORAGE_ACCESS_KEY=${AZURE_STORAGE_ACCESS_KEY} \
    --conf spark.executorEnv.AZURE_BLOB_CONTAINER=${AZURE_BLOB_CONTAINER} \
\
    --conf spark.kubernetes.driverEnv.PARTNER_BOS_REGION=${PARTNER_BOS_REGION} \
    --conf spark.kubernetes.driverEnv.PARTNER_BOS_BUCKET=${PARTNER_BOS_BUCKET} \
    --conf spark.kubernetes.driverEnv.PARTNER_BOS_ACCESS=${PARTNER_BOS_ACCESS} \
    --conf spark.kubernetes.driverEnv.PARTNER_BOS_SECRET=${PARTNER_BOS_SECRET} \
    --conf spark.executorEnv.PARTNER_BOS_REGION=${PARTNER_BOS_REGION} \
    --conf spark.executorEnv.PARTNER_BOS_BUCKET=${PARTNER_BOS_BUCKET} \
    --conf spark.executorEnv.PARTNER_BOS_ACCESS=${PARTNER_BOS_ACCESS} \
    --conf spark.executorEnv.PARTNER_BOS_SECRET=${PARTNER_BOS_SECRET} \
\
    "${JOB_FILE}" --running_mode=PROD $@
