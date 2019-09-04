#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
conda activate fuel-py36-pyro
export PYTHONPATH=$(python -m site --user-site):/apollo/modules/data/fuel
python /apollo/modules/data/fuel/fueling/control/dynamic_model/gp_regression/main.py
