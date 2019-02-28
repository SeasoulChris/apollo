#!/usr/bin/env bash

SERVICE=$1
PORT=$2
NAMESPACE=$3

function usage() {
  echo "$0 <SERVICE> <PORT> [NAMESPACE=default]"
  echo "Available services are:"
  kubectl get svc
}

if [ -z "${SERVICE}" ]; then
  usage
elif [ -z "${PORT}" ]; then
  usage
else
  if [ -z "${NAMESPACE}" ]; then
    NAMESPACE=default
  fi
  echo "Please visit http://localhost:8001/api/v1/namespaces/${NAMESPACE}/services/http:${SERVICE}:${PORT}/proxy/"
  kubectl proxy
fi
