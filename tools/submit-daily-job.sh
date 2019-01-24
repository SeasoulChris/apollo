#!/usr/bin/env bash
#
# Prerequisite:
#   spark-submit:
#       sudo pip install pyspark==2.4.0
#
#   JDK 8?:
#       sudo apt-get install openjdk-8-jdk
#       sudo update-alternatives --config java

cd "$( dirname "${BASH_SOURCE[0]}" )"
source env.sh

WORKERS=3

spark-submit \
    --master k8s://${K8S} \
    --deploy-mode cluster \
    --conf spark.executor.instances=${WORKERS} \
    --conf spark.kubernetes.container.image=${REPO}:${TAG} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.glusterfs.mount.path=/mnt/nfs \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.glusterfs.options.claimName=glusterfs-pvc-1t \
    /apollo/modules/data/fuel/jobs/daily.py
