#!/usr/bin/env bash

# Go to apollo-fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/../../.."

set -e
set -x

# Make sure you are calling run_prod() instead of run_test()!
# Feature extraction.
export PYTHONPATH=/apollo/modules/data/fuel:/home/$USER/.conda/envs/fuel-py36-pyro/bin/python:/apollo/py_proto:/apollo/modules/tools:/apollo/bazel-bin/cyber/py_wrapper:/apollo/cyber/python:/apollo/bazel-bin/cyber/py_wrapper:/apollo/cyber/python


python /apollo/modules/data/fuel/fueling/control/dynamic_model/gp_regression/main.py
