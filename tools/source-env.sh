#!/usr/bin/env bash
# Run with `source`.

CONDA_ENV=$1

# Setup Apollo.
source /apollo/scripts/apollo_base.sh
# Setup Python.
source /usr/local/miniconda/bin/activate "${CONDA_ENV}"

# Remove all system py27 paths.
OLD_PYTHONPATH="${PYTHONPATH}"
PYTHONPATH="/apollo/modules/data/fuel:/apollo/py_proto"
IFS=':' read -ra PARTS <<< "${OLD_PYTHONPATH}"
for PART in "${PARTS[@]}"; do
  if [[ "${PART}" != *"python2.7"* ]]; then
    PYTHONPATH="${PYTHONPATH}:${PART}"
  fi
done
export PYTHONPATH="${PYTHONPATH}"

# Remove system torch so we can access that in conda env.
LD_LIBRARY_PATH_BLACKLIST=(
    ":/usr/local/apollo/libtorch/lib"
    ":/usr/local/apollo/libtorch_gpu/lib"
)
for i in "${LD_LIBRARY_PATH_BLACKLIST[@]}"; do
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH/${i}/}"
done
