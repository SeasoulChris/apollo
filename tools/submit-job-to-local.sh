#!/usr/bin/env bash

# Quick check.
if [ ! -d "./fueling" ]; then
  echo "You must run from apollo-fuel root folder."
  exit 1
fi
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "${BASH_SOURCE[0]} [options] <job.py> [job-gflags]"
  echo ""
  echo "Options are:"
  grep "^    --" "${BASH_SOURCE[0]}" | grep ')' | sed 's/)//' | sed 's/|/,/'
  exit 0
fi

# Default value for configurable arguments.
JOB_FILE=""
CONDA_ENV="fuel-py36"
EXECUTOR_CORES=2
LOG_VERBOSITY="INFO"

while [ $# -gt 0 ]; do
  case "$1" in
    --env|-e)        # Conda environment: "-e fuel-py36"
      shift
      CONDA_ENV=$1
      ;;
    --cpu|-c)        # CPU count per worker: "-c 2"
      shift
      EXECUTOR_CORES=$1
      ;;
    --verbosity|-v)  # Log verbosity: "-v INFO", or DEBUG, WARNING, ERROR, FATAL.
      shift
      LOG_VERBOSITY=$1
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

source tools/source-env.sh ${CONDA_ENV}
export LOG_VERBOSITY=${LOG_VERBOSITY}

spark-submit --master "local[${EXECUTOR_CORES}]" "${JOB_FILE}" --running_mode=TEST $@ 2>&1 | \
    grep -v ' INFO '
