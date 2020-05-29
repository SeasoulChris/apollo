#!/usr/bin/env bash
# CI source bashrc.

export PATH=${PATH}:/apollo/scripts:/usr/local/miniconda/bin

if [ -e "/apollo/scripts/apollo_base.sh" ]; then
  source /apollo/scripts/apollo_base.sh
fi

ulimit -c unlimited

source /usr/local/miniconda/bin/activate fuel
source /usr/local/lib/bazel/bin/bazel-complete.bash

bash /fuel/tools/check.sh
