#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
unset PYTHONPATH
source activate fuel-py36-pyro
export PYTHONPATH=/apollo/modules/data/fuel:${PYTHONPATH}

# These is another version of torch installed in /usr/local/apollo/libtorch_gpu.
# We have to remove it from the LD_LIBRARY_PATH so we can use the one in our
# conda env.
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

python /apollo/modules/data/fuel/fueling/control/dynamic_model/gp_regression/main.py
