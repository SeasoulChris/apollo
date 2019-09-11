#!/usr/bin/env bash

# Jobs that run once everyday at 1:00 a.m.
# Crontab example: 0 1 * * * /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# The bag-to-record job is code-freezed.
JOB="/apollo/modules/data/fuel/fueling/data/pipelines/bag_to_record.py"
./tools/submit-job-to-k8s.sh --workers 10 --memory 24g --disk 800 \
    --image "hub.baidubce.com/apollo/spark:ubuntu-14.04_spark-2.4.0" \
    --fueling "/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip" \
    ${JOB}

# Job: Daily jobs.
JOB="fueling/daily_jobs.py"
./tools/submit-job-to-k8s.sh --workers 10 --memory 24g --disk 800 ${JOB} \
    --generate_small_records_of_last_n_days=30 \
    --index_records_of_last_n_days=30

# Job: Control profiling.
JOB="fueling/profiling/control_profiling_metrics.py"
./tools/submit-job-to-k8s.sh --workers 10 --memory 24g ${JOB}
JOB="fueling/profiling/control_profiling_visualization.py"
CONDA_ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --workers 10 --memory 24g -e ${CONDA_ENV} ${JOB}
