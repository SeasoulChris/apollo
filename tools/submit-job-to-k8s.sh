#!/usr/bin/env bash
#
# Prerequisite:
#   spark-submit:
#       sudo pip install pyspark==2.4.0
#
#   JDK 8?:
#       sudo apt-get install openjdk-8-jdk
#       sudo update-alternatives --config java

LOCAL_JOB="$1"

# Update config.
IMAGE=apolloauto/spark:xxx
K8S=https://x.x.x.x:6443
WORKERS=5

set -x
set -e

# Upload job files to BOS.
TIME=$(date +%Y%m%d-%H%M)
JOB_PATH="/mnt/bos/modules/data/jobs"
JOB_FILE="${JOB_PATH}/${TIME}_${USER}_$(basename ${LOCAL_JOB})"
FUELING_PKG="${JOB_PATH}/${TIME}_${USER}_fueling.zip"

sudo mkdir -p "${JOB_PATH}"
sudo cp "${LOCAL_JOB}" "${JOB_FILE}"

pushd "$( dirname "${BASH_SOURCE[0]}" )/.."
  rm -fr ./fueling.zip && \
  zip -r ./fueling.zip ./fueling && \
  sudo cp ./fueling.zip "${FUELING_PKG}"
popd

# Submit job with fueling package.
spark-submit \
    --master k8s://${K8S} \
    --deploy-mode cluster \
    --conf spark.executor.instances=${WORKERS} \
    --conf spark.kubernetes.container.image=${REPO}:${TAG} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --py-files "${FUELING_PKG}" \
    ${JOB_FILE}
