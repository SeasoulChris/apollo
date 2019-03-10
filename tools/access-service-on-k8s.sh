#!/usr/bin/env bash

PORT=$1
NAMESPACE=$2

function usage() {
  echo "$0 <PORT> [NAMESPACE=default]"
  exit 0
}

if [ -z "${PORT}" ]; then
  usage
else
  if [ -z "${NAMESPACE}" ]; then
    NAMESPACE=default
  fi

  SERVICES=$(kubectl get svc | grep "${PORT}/" | awk '{print $1}')
  if [ -z "${SERVICES}" ]; then
    echo "Cannot found any service with port ${PORT}"
  else
    echo "Available services:"
    while read -r service; do
      echo "*    http://localhost:8001/api/v1/namespaces/${NAMESPACE}/services/http:${service}:${PORT}/proxy/"
    done <<< "${SERVICES}"
    kubectl proxy
  fi
fi
