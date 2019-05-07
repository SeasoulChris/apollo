#!/usr/bin/env bash

# Default value for configurable arguments.
JOB_FILE=""
CONDA_ENV="fuel-py27-cyber"
EXECUTOR_CORES=2

while [ $# -gt 0 ]; do
  case "$1" in
    --env|-e)
      shift
      CONDA_ENV=$1
      ;;
    --cpu|-c)
      shift
      EXECUTOR_CORES=$1
      ;;
    *)
      if [ -f "$1" ]; then
        JOB_FILE=$1
        shift
        break
      else
        echo -e "$1: Unknown option or file not exists."
        exit 1
      fi
      ;;
  esac
  shift
done

if [ -z "${JOB_FILE}" ]; then
  echo "No job specified."
  exit 1
fi

source /usr/local/miniconda/bin/activate ${CONDA_ENV}
source /apollo/scripts/apollo_base.sh

spark-submit --master "local[${EXECUTOR_CORES}]" "${JOB_FILE}" --running_mode=TEST $@
