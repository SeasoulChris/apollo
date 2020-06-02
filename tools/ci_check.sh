#!/usr/bin/env bash

export PATH=${PATH}:/apollo/scripts:/usr/local/miniconda/bin

if [ -e "/apollo/scripts/apollo_base.sh" ]; then
  source /apollo/scripts/apollo_base.sh
fi

source /usr/local/miniconda/bin/activate fuel

bash /fuel/tools/check.sh
