#!/usr/bin/env bash

export PATH=${PATH}:/apollo/scripts:/usr/local/miniconda/bin

source /apollo/scripts/apollo_base.sh
source /usr/local/miniconda/bin/activate fuel
bash /fuel/tools/check.sh
