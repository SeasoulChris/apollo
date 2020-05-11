#!/bin/bash

function print_usage() {
    echo 'Usage:
    ./main.sh -c [ bce-platform | az-staging ] -a [ init | build | run -f <CONFIG_FILE> -w <NUMBER_OF_WORKERS> | stop ]
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

  set -ex
  cd $( dirname "${BASH_SOURCE[0]}" )/../..

  # check if the config file is existing
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "$FILE not found."
    exit 1
  fi

  STUDY_NAME="supertuner-$USER-$(date +%Y%m%d_%H%M%S)"

  # upload config file to /mnt/bos
  # TODO: save config file under tuner_storage_dir (/mnt/bos/autotuner/$STUDY_NAME). Having an issue to access it as on 5/11/2020
  CONFIG_FILE_DEST="modules/autotuner/$STUDY_NAME/$(basename "${CONFIG_FILE}")"
  apps/local/bos_fstool --src=${CONFIG_FILE} --dst=${CONFIG_FILE_DEST}

  # spin off worker(s)
  RUN_FILE="${DEPLOY_DIR}/optuna_worker.yaml"
  IMG="${DEST_REPO}/${IMAGE}"
  sed -i "s|__IMG__|$IMG|g;s|__CLUSTER__|$CLUSTER|g;s|__CONFIG_FILE__|/mnt/bos/$CONFIG_FILE_DEST|g; \
          s|__ROLE__|$USER|;s|__STUDY_NAME__|$STUDY_NAME|g;s|__WORKER_COUNT__|$WORKER_COUNT|g" \
        $RUN_FILE
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
      DEST_REPO="simengineregistry.azurecr.io"
      K8S_NAMESPACE="default"
      ;;
    bce-platform)
      DEST_REPO="hub.baidubce.com/apollofuel/autotuner"
      K8S_NAMESPACE="autotuner"
      ;;
    *)
      print_usage
      ;;
  esac
}

function main() {
  CLUSTER=''
  ACTION=''
  CONFIG_FILE=''
  WORKER_COUNT=1

  # read options
  while getopts 'c:a:f:w:' flag; do
    case "${flag}" in
      c)
        CLUSTER="${OPTARG}"
      ;;
      a)
        ACTION="${OPTARG}"
      ;;
      f)
        CONFIG_FILE="${OPTARG}"
      ;;
      w)
        WORKER_COUNT="${OPTARG}"
      ;;
      *)
        print_usage
        exit 1
      ;;
    esac
  done

  # validate inputs
  if [[ -z "${CLUSTER}" || -z "${ACTION}" ]] || [[ "${ACTION}" = "run" && -z "${CONFIG_FILE}" ]]; then
    print_usage
    exit 1
  fi


  check_cluster
  init_settings

  case "$ACTION" in
    init)
      init_environment
      ;;
    build)
      # note: need to login to docker repo before executing this option
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
      exit 1
      ;;
  esac
}

main "$@"
