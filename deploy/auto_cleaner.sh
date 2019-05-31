#!/usr/bin/env bash

# Run once everyday hour.
# Crontab example: 0 */1 * * * /this/script.sh

set -e

function DeletePod() {
  DRIVER_POD_STATUS=$1
  CUT_TIME=$2

  CUT_TIMESTAMP=$(date -d "${CUT_TIME}" +%s%3N)
  kubectl get pods | grep '\-driver' | grep "${DRIVER_POD_STATUS}" | awk '{print $1}' | \
  while read -r DRIVER_POD; do
    DRIVER_TIMESTAMP=$(echo "${DRIVER_POD}" | awk -F- '{print $(NF-1)}')
    if [ "${DRIVER_TIMESTAMP}" -lt "${CUT_TIMESTAMP}" ]; then
      echo "Delete ${DRIVER_POD_STATUS} pod ${DRIVER_POD}"
      kubectl delete pods "${DRIVER_POD}"
    fi
  done
}

DeletePod 'Completed' '12 hours ago'
DeletePod 'Error'     '24 hours ago'
