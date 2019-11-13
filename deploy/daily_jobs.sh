#!/usr/bin/env bash

# Jobs that run once everyday at 1:00 a.m.
# Crontab example: @daily /this/script.sh

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# The bag-to-record job is code-freezed.
JOB="/apollo/modules/data/fuel/fueling/data/pipelines/bag_to_record.py"
FUEL_CLIENT_IMAGE="apolloauto/fuel-client:20190821_1718" ./tools/submit-job-to-k8s.sh \
    --workers 10 --memory 20g --disk 500 \
    --image "hub.baidubce.com/apollo/spark:ubuntu-14.04_spark-2.4.0" \
    --fueling "/mnt/bos/modules/data/jobs/deploy/fueling-latest.zip" \
    ${JOB}

# Job: Daily jobs.
SUBMITTER="./tools/submit-job-to-k8s.py --env=fuel-py36 --workers=10 --memory=24"
${SUBMITTER} --main="fueling/daily_jobs.py" --disk=800
