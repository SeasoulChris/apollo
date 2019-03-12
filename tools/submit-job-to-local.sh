#!/usr/bin/env bash

# Default config.
JOB_FILE=""
CONDA_ENV="fuel-py27-cyber"
EXECUTOR_CORES=4

while [ $# -gt 0 ]; do
    case "$1" in
    --env)
        shift
        CONDA_ENV=$1
        ;;
    --job)
        shift
        JOB_FILE=$1
        ;;
    --worker-cpu)
        shift
        EXECUTOR_CORES=$1
        ;;
    *)
        echo -e "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

source /usr/local/miniconda2/bin/activate ${CONDA_ENV}
source /apollo/scripts/apollo_base.sh

spark-submit --master "local[${EXECUTOR_CORES}]" "${JOB_FILE}"
