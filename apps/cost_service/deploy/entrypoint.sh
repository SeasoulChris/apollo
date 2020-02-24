#! /bin/bash

function print_usage() {
    echo 'Usage:
    ./entrypoint.sh [ bce-platform | bce-staging | az-staging | local ]
    '
}

function mount_bos() {
  # Mount bosfs data
  # Check https://cloud.baidu.com/doc/BOS/BOSCLI.html#.E5.AE.89.E8.A3.85RPM.E5.8C.85
  mkdir -p /mnt/bos
  bosfs apollo-platform /mnt/bos -o allow_other,logfile=/tmp/bos.log,endpoint=http://bj.bcebos.com,ak=5bd2738d9d6e4f0ea14405da17285971,sk=b624fc868051475e91fd507e46c4ce56
  if [ $? -ne 0 ]; then
    echo 'Failed to mount /mnt/bos!'
    exit 1
  fi
}

function start_service() {
  source /home/libs/bash.rc

  LOG_DIR=/fuel/fueling/learning/autotuner/log
  mkdir -p $LOG_DIR

  bazel run //fueling/learning/autotuner/cost_computation:cost_computation_server -- \
      --running_mode=$MODE \
      --sim_service_url="${SIM_SERVICE_URL}" \
      ${ADDITIONAL_FLAGS} \
      2>&1 | tee $LOG_DIR/cost_service.log; test ${PIPESTATUS[0]} -eq 0
}

function main() {
  if [ $# -ne 1 ]; then
      print_usage
      exit 1
  fi

  local cluster=$1
  case "$cluster" in
    local)
      MODE=TEST
      SIM_SERVICE_URL="localhost:50051"
      ;;
    az-staging)
      MODE=TEST
      SIM_SERVICE_URL="simservice:50051"
      ;;
    bce-staging)
      MODE=TEST
      SIM_SERVICE_URL="simservice:50051"
      ;;
    bce-platform)
      MODE=PROD
      SIM_SERVICE_URL="simservice:50051"
      ADDITIONAL_FLAGS="--wait --workers=5 \
          --spark_submitter_service_url=http://spark-submitter-service:8000"
      ;;
    *)
      print_usage
      ;;
  esac

  mount_bos
  start_service

}

main "$@"
