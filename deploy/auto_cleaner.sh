#!/usr/bin/env bash

# Run once everyday hour.
# Crontab example: 0 */1 * * * /this/script.sh

set -e

CUT_TIMESTAMP=$(date -d '12 hours ago' +%s%3N)

kubectl get pods | grep '\-driver' | grep -e 'Completed' -e 'Error' | awk '{print $1}' | \
while read -r DRIVER_POD; do
  DRIVER_TIMESTAMP=$(echo "${DRIVER_POD}" | awk -F- '{print $(NF-1)}')
  if [ "${DRIVER_TIMESTAMP}" -lt "${CUT_TIMESTAMP}" ]; then
    echo "Delete pod ${DRIVER_POD}"
    kubectl delete pods "${DRIVER_POD}"
  fi
done
