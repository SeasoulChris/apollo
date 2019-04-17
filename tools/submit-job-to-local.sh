#!/usr/bin/env bash

# Default value for configurable arguments.
JOB_FILE=""
FLAGFILE="fueling/common/flagfile/local_job.flag"
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
    --flagfile|-f)
      shift
      # It must start from apollo-fuel root, such as 'fueling/...'.
      FLAGFILE=$1
      ;;
    *)
      if [ -f "$1" ]; then
        JOB_FILE=$1
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

source /usr/local/miniconda2/bin/activate ${CONDA_ENV}
source /apollo/scripts/apollo_base.sh

FLAGFILE="${FLAGFILE}" spark-submit --master "local[${EXECUTOR_CORES}]" "${JOB_FILE}"
