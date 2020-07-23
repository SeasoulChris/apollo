#!/bin/bash

source ./common.sh

function print_usage() {
    echo "Usage:
    ./cost_service.sh [ init | deploy ] [ $(join_by ' | ' ${VALID_CLUSTERS[@]}) ]
    "
}

function scale_deployment() {
  kubectl scale deployment costservice-deployment -n $K8S_NAMESPACE --replicas=$REPLICA
}

function deploy() {
  echo "Deleting the current deployment"
  kubectl delete deployment costservice-deployment -n $K8S_NAMESPACE

  set -e  
  echo "Deploying..."
  DEPLOY_FILE="${DEPLOY_DIR}/costservice_deployment.yaml"
  IMG="${DEST_REPO}/${IMAGE_NAME}"
  local deploy_api_version=$(get_k8s_api_version "deploy")
  sed -i "s|__DEPLOY_API_VERSION__|$deploy_api_version|g;s|__IMG__|$IMG|g;s|__CLUSTER__|$CLUSTER|g;s|__NAMESPACE__|$K8S_NAMESPACE|g" $DEPLOY_FILE
  kubectl create -f $DEPLOY_FILE
  git checkout -- $DEPLOY_FILE
}

function init_environment() {
  if [ "$K8S_NAMESPACE" != "default" ]; then
    kubectl create namespace $K8S_NAMESPACE
    kubectl create secret -n $K8S_NAMESPACE docker-registry regsecret --docker-server=hub.baidubce.com \
      --docker-username=apollofuel \
      --docker-password=apollo@2017
  fi

  SERVICE_FILE="${DEPLOY_DIR}/costservice_service.yaml"
  sed -i "s|__NAMESPACE__|$K8S_NAMESPACE|g" $SERVICE_FILE
  kubectl create -f $SERVICE_FILE
  git checkout -- $SERVICE_FILE

  echo "Done creating namespace and services, remember to create secrets (see instruction from wiki Credentials page)"
}


function main() {
  if [ $# -ne 2 ]; then
    print_usage
    exit 1
  fi

  ACTION=$1
  CLUSTER=$2
  check_cluster $CLUSTER
  init_settings

  case "$ACTION" in
    init)
      init_environment
      ;;
    deploy)
      deploy
      scale_deployment
      ;;
    *)
      print_usage
      ;;
  esac
}

main "$@"
