#!/usr/bin/env bash
#
# Prerequisite:
#   spark-submit:
#       sudo pip install pyspark==2.4.0
#
#   JDK 8?:
#       sudo apt-get install openjdk-8-jdk
#       sudo update-alternatives --config java

cd "$( dirname "${BASH_SOURCE[0]}" )/.."
source tools/base.sh

WORKERS=1

# Make fueling package to submit job.
rm -fr ./fueling.zip && zip -r ./fueling.zip ./fueling

spark-submit \
    --master k8s://${K8S} \
    --deploy-mode cluster \
    --conf spark.executor.instances=${WORKERS} \
    --conf spark.kubernetes.container.image=${REPO}:${TAG} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --py-files ./fueling.zip \
    $1
