#!/bin/bash

function print_usage() {
    echo 'Usage:
    ./main.sh [ build | deploy ] [ bce-platform | bce-staging | az-staging ]
    '
}

function scale_deployment() {
  kubectl scale deployment costservice-deployment --replicas=$REPLICA
}

function deploy() {
  echo "Deleting the current deployment"
  kubectl delete service costservice
  kubectl delete deployment costservice-deployment

  set -e  
  echo "Deploying..."
  DEPLOY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/deploy"
  sed -i "s|__IMG__|$DEST_REPO|g;s|__CLUSTER__|$CLUSTER|g" "${DEPLOY_DIR}"/costservice.yaml
  kubectl create -f "${DEPLOY_DIR}"/costservice.yaml
  git checkout -- "${DEPLOY_DIR}"/costservice.yaml
}

function build_and_push() {
  echo 'Start building cost_service image ...'
  cd $( dirname "${BASH_SOURCE[0]}" )/../..

  set -e
  set -x
  docker build -t ${IMAGE} --network host -f apps/cost_service/docker/Dockerfile .

  echo 'Start pushing cost_service image ...'
  docker tag ${IMAGE} ${DEST_REPO}
  docker push ${DEST_REPO}
}

function check_cluster() {
  local current_cluster=`kubectl config current-context`
  echo "Current cluster is ${current_cluster}"

  if [ "${current_cluster}" != "${CLUSTER}" ]; then
    echo "Current cluster isn't the deploy one, please switch your cluster."
    exit 1
  fi
}

function init() {
  IMAGE="cost_service:latest"
  case "$CLUSTER" in
    az-staging)
      docker login simengineregistry.azurecr.io -u SImEngineRegistry -p dBHhbaaj3gBlFMpIDLWDwdeaUzLrsIL/
      DEST_REPO="simengineregistry.azurecr.io/${IMAGE}"
      REPLICA=1
      ;;
    bce-staging)
      docker login hub.baidubce.com -u apollo -p apollo@2017
      DEST_REPO="hub.baidubce.com/apollo/autotuner_staging/${IMAGE}"
      REPLICA=1
      ;;
    bce-platform)
      docker login hub.baidubce.com -u apollo -p apollo@2017
      DEST_REPO="hub.baidubce.com/apollo/autotuner/${IMAGE}"
      REPLICA=2
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

  ACTION=$1
  CLUSTER=$2
  check_cluster
  init

  case "$ACTION" in
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
