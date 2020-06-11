#!/bin/bash

VALID_CLUSTERS=( "az-staging" "bce-debug" "bce-staing" "bce-platform" )

function join_by() {
  local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}";
}

function check_cluster() {
  local desired_cluster=$1
  local current_cluster=`kubectl config current-context`
  echo "Current cluster is ${current_cluster}"

  if [ "${current_cluster}" = "kubernetes-admin@kubernetes" ]; then
    current_cluster="bce-platform"
  fi

  if [ "${current_cluster}" != "${desired_cluster}" ]; then
    echo "Current cluster isn't the deploy one ($desired_cluster), please switch your cluster."
    exit 1
  fi
}

function init_settings() {
  IMAGE="cost_service"
  DEPLOY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/deploy"
  case "$CLUSTER" in
    az-staging)
      DEST_REPO="simengineregistry.azurecr.io"
      K8S_NAMESPACE="default"
      REPLICA=1
      ;;
    bce-debug)
      DEST_REPO="hub.baidubce.com/apollofuel/autotuner_staging"
      K8S_NAMESPACE="autotuner-debug"
      REPLICA=2
      ;;
    bce-staging)
      DEST_REPO="hub.baidubce.com/apollofuel/autotuner_staging"
      K8S_NAMESPACE="default"
      REPLICA=1
      ;;
    bce-platform)
      DEST_REPO="hub.baidubce.com/apollofuel/autotuner"
      K8S_NAMESPACE="autotuner"
      REPLICA=2
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
}
