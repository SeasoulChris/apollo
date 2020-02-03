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
EXECUTOR_CORES=2
LOG_VERBOSITY="INFO"

while [ $# -gt 0 ]; do
  case "$1" in
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

source /usr/local/miniconda/bin/activate fuel
export LOG_VERBOSITY=${LOG_VERBOSITY}

set -ex
TARGET_FILE=$(bazel query ${JOB_FILE})
TARGET=$(bazel query "attr('srcs', $TARGET_FILE, ${TARGET_FILE//:*/}:*)")

./v2/build.sh ${TARGET}
BUILT_TARGET="bazel-bin/${JOB_FILE%.*}_main.py"

spark-submit --master "local[${EXECUTOR_CORES}]" "${BUILT_TARGET}" --running_mode=TEST $@ 2>&1 | \
    grep -v ' INFO '
