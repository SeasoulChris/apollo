#!/bin/bash

function print_usage() {
    echo 'Usage:
    ./main.sh [ bce-platform | az-staging ] [ init | build | run | stop ]
    '
}

function scale_deployment() {
  kubectl scale deployment costservice-deployment -n $K8S_NAMESPACE --replicas=$REPLICA
}

function delete_worker() {
  kubectl delete job.batch/$USER-optuna-worker
}

function run() {
  delete_worker

  RUN_FILE="${DEPLOY_DIR}/optuna_worker.yaml"
  IMG="${DEST_REPO}/${IMAGE}"
  sed -i "s|__IMG__|$IMG|g;s|__CLUSTER__|$CLUSTER|g;s|__NAMESPACE__|$K8S_NAMESPACE|g;s|__ROLE__|$USER|" $RUN_FILE
  kubectl apply -f $RUN_FILE
  git checkout -- $RUN_FILE
}

function build_and_push() {
  # NOTE: this will be depreicated soon as optuna and cost_service can share the same image
  echo 'Building optuna image ...'
  cd $( dirname "${BASH_SOURCE[0]}" )/../..

  set -ex
  docker build -t ${IMAGE} --network host -f apps/optuna/docker/Dockerfile .

  echo 'Start pushing optuna image ...'
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

  SERVICE_FILE="${DEPLOY_DIR}/optuna_init.yaml"
  sed -i "s|__NAMESPACE__|$K8S_NAMESPACE|g" $SERVICE_FILE
  kubectl create -f $SERVICE_FILE
  git checkout -- $SERVICE_FILE
}

function init_settings() {
  IMAGE="optuna_worker"
  DEPLOY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/deploy"
  case "$CLUSTER" in
    az-staging)
      docker login simengineregistry.azurecr.io -u SImEngineRegistry -p dBHhbaaj3gBlFMpIDLWDwdeaUzLrsIL/
      DEST_REPO="simengineregistry.azurecr.io"
      K8S_NAMESPACE="default"
      ;;
    bce-platform)
      docker login hub.baidubce.com -u apollofuel -p apollo@2017
      DEST_REPO="hub.baidubce.com/apollofuel/autotuner"
      K8S_NAMESPACE="autotuner"
      ;;
    *)
      print_usage
      ;;
  esac
}

function main() {
  if [ $# -ne 2 ]; then
    print_usage
    exit 1
  fi

  CLUSTER=$1
  ACTION=$2
  check_cluster
  init_settings

  case "$ACTION" in
    init)
      init_environment
      ;;
    build)
      build_and_push
      ;;
    run)
      run
      ;;
    stop)
      delete_worker
      ;;
    *)
      print_usage
      ;;
  esac
}

main "$@"
