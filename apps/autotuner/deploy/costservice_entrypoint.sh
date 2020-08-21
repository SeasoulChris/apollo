#! /bin/bash

function print_usage() {
    echo 'Usage:
    ./entrypoint.sh [ bce-platform | bce-staging | bce-debug | az-staging | local ]
    '
}

function mount_bos() {
  # Mount bosfs data
  # Check https://cloud.baidu.com/doc/BOS/BOSCLI.html#.E5.AE.89.E8.A3.85RPM.E5.8C.85
  RW_MOUNT="/mnt/bos-rw"
  RO_MOUNT="/mnt/bos-ro"
  EXPOSE_MOUNT="/mnt/bos"
  mkdir -p "${RW_MOUNT}" "${RO_MOUNT}" "${EXPOSE_MOUNT}"

  # bucket, ak, and sk values are from bos-secret
  bosfs "${bucket}" "${RW_MOUNT}" -o logfile=/tmp/bos-rw.log,endpoint=http://bj.bcebos.com,ak=${ak},sk=${sk}
  bosfs "${bucket}" "${RO_MOUNT}" -o ro,logfile=/tmp/bos-ro.log,endpoint=http://bj.bcebos.com,ak=${ak},sk=${sk}
  if [ $? -ne 0 ]; then
    echo 'Failed to mount /mnt/bos!'
    exit 1
  fi

  READONLY_PATHS=( "modules" )
  for subpath in "${READONLY_PATHS[@]}"; do
    ln -s "${RO_MOUNT}/${subpath}" "${EXPOSE_MOUNT}/${subpath}"
  done

  READWRITE_PATHS=( "autotuner" )
  for subpath in "${READWRITE_PATHS[@]}"; do
    ln -s "${RW_MOUNT}/${subpath}" "${EXPOSE_MOUNT}/${subpath}"
  done
}

function start_service() {
  source /home/libs/bash.rc
  source /apollo/scripts/apollo_base.sh

  echo "GITHUB_VERSION: $GITHUB_VERSION"

  # GRPC setting
  export GRPC_VERBOSITY=INFO
  # uncomment me to debug grpc connection issues
  # export GRPC_TRACE=call_error,client_channel_call,connectivity_state,server_channel

  # A temp workaround until a new fuel-client image has this package.
  pip install --no-cache-dir grpcio-health-checking

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
    bce-debug)
      SUBMIT_SPARK_JOB_TO_K8S="true"
      SIM_SERVICE_URL="simservice.autotuner-debug.svc.cluster.local:50051"
      ADDITIONAL_FLAGS="--spark_submitter_service_url=http://spark-submitter-service.default.svc.cluster.local:8000"
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
