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

  STUDY_NAME="autotuner-$(date +%Y%m%d_%H%M)"
  LOG_DIR="/mnt/bos/autotuner/optuna/$STUDY_NAME"
  mkdir -p $LOG_DIR

  LOG_NAME=$LOG_DIR/optuna_$MY_POD_NAME.log
  echo "Saving log to $LOG_NAME"

  # TODO(vivian): allow user to set worker_number and tuner_param_config_filename
  bazel run //fueling/learning/autotuner/tuner:optuna_super_tuner -- \
    --cost_computation_service_url=$COST_SERVICE_URL \
    --study_storage_url=$STUDY_STORAGE_URL \
    --tuner_param_config_filename=fueling/learning/autotuner/config/mrac_tuner_param_config.pb.txt \
    --study_name=$STUDY_NAME \
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
