#!/usr/bin/env bash

# Run once everyday hour.
# Crontab example: @hourly /this/script.sh

set -e

DRIVERS=$(kubectl get pods | grep '\-driver')
COMPLETED_DRIVERS=$(echo "${DRIVERS}" | grep Completed | awk '{print $1}')
ERROR_DRIVERS=$(echo "${DRIVERS}" | grep Error | awk '{print $1}')

function DeletePod() {
  CUT_TIME=$1
  CUT_TIMESTAMP=$(date -d "${CUT_TIME}" +%s%3N)
  shift

  while [ $# -gt 0 ]; do
    DRIVER_POD=$1
    DRIVER_TIMESTAMP=$(echo "${DRIVER_POD}" | awk -F- '{print $(NF-1)}')
    if [ "${DRIVER_TIMESTAMP}" -lt "${CUT_TIMESTAMP}" ]; then
      DRIVER_TIMESTAMP_SEC=$(echo "${DRIVER_TIMESTAMP}/1000" | bc)
      DRIVER_STARTING_TIME=$(date -d @${DRIVER_TIMESTAMP_SEC} "+%F %T")
      echo "Deleting pod [${DRIVER_STARTING_TIME}] ${DRIVER_POD} ..."
      kubectl delete pods "${DRIVER_POD}"
    fi
    shift
  done
}

DeletePod '12 hours ago' ${COMPLETED_DRIVERS}
DeletePod '24 hours ago' $(echo ${ERROR_DRIVERS} | grep -v prediction-app-performance-evaluation)
