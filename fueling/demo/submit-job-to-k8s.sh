#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

./tools/submit-job-to-k8s.sh --worker 1 --cpu 1 --memory 8g $@
