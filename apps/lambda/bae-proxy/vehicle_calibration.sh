#!/usr/bin/env bash

BASH_ARGS=$1
PY_ARGS=$2

export PATH=/usr/local/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FUELING="/mnt/bos/modules/data/jobs/deploy/20190531-133701_fueling.zip"
SUBMITER="/apollo/modules/data/fuel/tools/submit-job-to-k8s.sh --fueling ${FUELING} ${BASH_ARGS}"

${SUBMITER} -w 1 -c 1 -m 1g -d 1 /apollo/modules/data/fuel/fueling/demo/count-msg-by-channel.py ${PY_ARGS}
