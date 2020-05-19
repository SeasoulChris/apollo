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
  bosfs apollo-platform-fuel /mnt/bos -o allow_other,logfile=/tmp/bos.log,endpoint=http://bj.bcebos.com,ak=a07590cd90a54310ab68652c97b9bc21,sk=50ddc29c022c4a549973706678dfcddd
  if [ $? -ne 0 ]; then
    echo 'Failed to mount /mnt/bos!'
    exit 1
  fi
}

function start_service() {
  source /home/libs/bash.rc
  source /apollo/scripts/apollo_base.sh

  # GRPC setting
  export GRPC_VERBOSITY=INFO
  export GRPC_TRACE=client_channel_call,client_channel_routing,connectivity_state,server_channel

  LOG_DIR=/tmp/log
  mkdir -p $LOG_DIR

  bazel run //fueling/learning/autotuner/cost_computation:cost_computation_server -- \
      --cloud=${SUBMIT_SPARK_JOB_TO_K8S} \
      --sim_service_url=${SIM_SERVICE_URL} \
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
      SUBMIT_SPARK_JOB_TO_K8S="false"
      SIM_SERVICE_URL="localhost:50051"
      ;;
    az-staging)
      SUBMIT_SPARK_JOB_TO_K8S="false"
      SIM_SERVICE_URL="simservice:50051"
      ;;
    bce-staging)
      SUBMIT_SPARK_JOB_TO_K8S="false"
      SIM_SERVICE_URL="simservice:50051"
      ;;
    bce-platform)
      SUBMIT_SPARK_JOB_TO_K8S="true"
      SIM_SERVICE_URL="simservice.autotuner.svc.cluster.local:50051"
      ADDITIONAL_FLAGS="--spark_submitter_service_url=http://spark-submitter-service.default.svc.cluster.local:8000"
      ;;
    *)
      print_usage
      ;;
  esac

  mount_bos
  start_service

}

main "$@"
