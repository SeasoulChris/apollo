#! /bin/bash

function print_usage() {
    echo 'Usage:
    ./optuna_entrypoint.sh [ bce-platform | az-staging ]
    '
}

function mount_bos() {
  # Mount bosfs data
  # Check https://cloud.baidu.com/doc/BOS/BOSCLI.html#.E5.AE.89.E8.A3.85RPM.E5.8C.85
  mkdir -p /mnt/bos
  bosfs apollo-platform-fuel /mnt/bos -o allow_other,logfile=/tmp/bos.log,endpoint=http://bj.bcebos.com,ak=a07590cd90a54310ab68652c97b9bc21,sk=50ddc29c022c4a549973706678dfcddd
  if [ $? -ne 0 ]; then
    echo 'Failed to mount /mnt/bos!'
    exit 1
  fi
}

function start_service() {
  source /home/libs/bash.rc
  source /apollo/scripts/apollo_base.sh

  echo "GITHUB_VERSION: $GITHUB_VERSION"

  # GRPC settings
  export GRPC_VERBOSITY=INFO
  export GRPC_TRACE=call_error,client_channel_call,client_channel_routing,connectivity_state,server_channel

  echo "Study Name: $STUDY_NAME. Config File: $CONFIG_FILE. Worker Count: $WORKER_COUNT"
  # where images/logs/config are stored
  OUT_DIR="/mnt/bos/autotuner/$STUDY_NAME"
  LOG_DIR="$OUT_DIR/logs"
  mkdir -p $LOG_DIR

  LOG_NAME=$LOG_DIR/$MY_POD_NAME.log
  echo "Saving log to $LOG_NAME"

  bazel run //fueling/learning/autotuner/tuner:optuna_super_tuner -- \
    --running_role_postfix=$ROLE \
    --cost_computation_service_url=$COST_SERVICE_URL \
    --tuner_param_config_filename=$CONFIG_FILE \
    --tuner_storage_dir=$OUT_DIR \
    --study_storage_url=$STUDY_STORAGE_URL \
    --study_name=$STUDY_NAME \
    --n_coworkers=$WORKER_COUNT \
    2>&1 | tee $LOG_NAME; test ${PIPESTATUS[0]} -eq 0
}

function main() {
  if [ $# -ne 1 ]; then
      print_usage
      exit 1
  fi

  local cluster=$1
  case "$cluster" in
    az-staging)
      COST_SERVICE_URL="40.77.110.196:50052"
      STUDY_STORAGE_URL="40.77.100.63:5432"
      ;;
    bce-debug)
      COST_SERVICE_URL="180.76.111.129:50052"
      STUDY_STORAGE_URL="autotune-postgres.autotuner-debug.svc.cluster.local:5432"
      ;;
    bce-platform)
      COST_SERVICE_URL="180.76.242.157:50052"
      STUDY_STORAGE_URL="autotune-postgres.autotuner.svc.cluster.local:5432"
      ;;
    *)
      print_usage
      ;;
  esac
  
  mount_bos
  start_service
}

main "$@"
