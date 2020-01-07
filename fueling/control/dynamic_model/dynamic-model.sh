#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

JOB_ID=$(date +%Y-%m-%d-%H-%M)

#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

SUBMITTER="./tools/submit-job-to-k8s.py --workers=12 --cpu=6 --memory=60"
INPUT_DATA_PATH="modules/control/data/records"

JOB="fueling/control/dynamic_model/dynamic_model.py"
FLAGS="--input_data_path=${INPUT_DATA_PATH} --job_id=${JOB_ID}"
${SUBMITTER} --main=${JOB} --flags="${FLAGS}"
