#!/bin/bash

source ./common.sh

function print_usage() {
    echo "Usage:
    ./super_tuner.sh -c [ $(join_by ' | ' ${VALID_CLUSTERS[@]}) ] -a [ init | run -f <CONFIG_FILE> -w <NUMBER_OF_WORKERS> | stop ]
    "
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
  if [[ -z "${CLUSTER}" || -z "${ACTION}" ]] || [[ "${ACTION}" == "run" && -z "${CONFIG_FILE}" ]]; then
    print_usage
    exit 1
  fi

  check_cluster $CLUSTER
  init_settings

  case "$ACTION" in
    init)
      init_environment
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
