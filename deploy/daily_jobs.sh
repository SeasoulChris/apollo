#!/usr/bin/env bash

set -e

# Preapre: Goto fuel root, checkout latest code.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Job: Generate small records.
JOB="fueling/data/pipelines/generate-small-records.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g ${JOB}
JOB="fueling/data/pipelines/reorg-small-records.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g ${JOB}

# Job: Bags to records.
JOB="fueling/data/pipelines/bag-to-record.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g ${JOB}

# Job: Index records.
JOB="fueling/data/pipelines/index-records.py"
./tools/submit-job-to-k8s.sh --workers 16 --memory 24g ${JOB}

# Job: Control profiling.
JOB="fueling/control/control_profiling/control-profiling-metrics.py"
./tools/submit-job-to-k8s.sh ${JOB}
JOB="fueling/control/control_profiling/control-profiling-visualization.py"
CONDA_ENV="fuel-py27"
./tools/submit-job-to-k8s.sh -e ${CONDA_ENV} ${JOB}
