#!/usr/bin/env bash

function Run() {
  # Go to apollo-fuel root.
  cd "$( dirname "${BASH_SOURCE[0]}" )/.."
  # Fail on first error.
  set -e

  git pull

  # 1. Generate small records.
  JOB="fueling/data/pipelines/generate-small-records.py"
  ENV="fuel-py27-cyber"
  ./tools/submit-job-to-k8s.sh ${JOB} --env ${ENV} --workers 16 --cpu 1 --memory 20g
}

function Help() {
  echo "Add this to 'crontab -e':"
  #     m h
  echo "0 3 * * * $(realpath "${BASH_SOURCE[0]}") run"
}

if [ "$1" = "run" ]; then
  Run
else
  Help
fi
