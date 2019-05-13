#!/usr/bin/env bash

# Jobs that run once everyday at 1:00 a.m.
# Crontab example: 0 1 * * * /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Job: Generate small records.
JOB="fueling/data/pipelines/generate-small-records.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g --disk 500 ${JOB}
JOB="fueling/data/pipelines/reorg-small-records.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g --disk 200 ${JOB}

# Job: Bags to records.
JOB="fueling/data/pipelines/bag-to-record.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g --disk 500 ${JOB}

# Job: Index records.
JOB="fueling/data/pipelines/index-records.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g ${JOB}

# Job: Control profiling.
JOB="fueling/control/control_profiling/control-profiling-metrics.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g ${JOB}
JOB="fueling/control/control_profiling/control-profiling-visualization.py"
CONDA_ENV="fuel-py27"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g -e ${CONDA_ENV} ${JOB}
