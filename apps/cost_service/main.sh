#!/bin/bash

function print_usage() {
    echo 'Usage:
    ./main.sh [ init | build | deploy ] [ bce-platform | bce-staging | bce-debug | az-staging ]
    '
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
  IMG="${DEST_REPO}/${IMAGE}"
  sed -i "s|__IMG__|$IMG|g;s|__CLUSTER__|$CLUSTER|g;s|__NAMESPACE__|$K8S_NAMESPACE|g" $DEPLOY_FILE
  kubectl create -f $DEPLOY_FILE
  git checkout -- $DEPLOY_FILE
}

function build_and_push() {
  echo 'Building cost_service image ...'
  cd $( dirname "${BASH_SOURCE[0]}" )/../..

  set -ex
  docker build -t ${IMAGE} --network host -f apps/cost_service/docker/Dockerfile .

  echo 'Start pushing cost_service image ...'
  TAG="$(date +%Y%m%d_%H%M)"
  docker tag ${IMAGE} "${DEST_REPO}/${IMAGE}:${TAG}"
  docker push "${DEST_REPO}/${IMAGE}:${TAG}"

  TAG="latest"
  docker tag ${IMAGE} "${DEST_REPO}/${IMAGE}:${TAG}"
  docker push "${DEST_REPO}/${IMAGE}:${TAG}"
}

function check_cluster() {
  local current_cluster=`kubectl config current-context`
  echo "Current cluster is ${current_cluster}"

  if [ "${current_cluster}" != "${CLUSTER}" ]; then
    echo "Current cluster isn't the deploy one, please switch your cluster."
    exit 1
  fi
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

function main() {
  if [ $# -ne 2 ]; then
    print_usage
    exit 1
  fi

  ACTION=$1
  CLUSTER=$2
  check_cluster
  init_settings

  case "$ACTION" in
    init)
      init_environment
      ;;
    build)
      build_and_push
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
