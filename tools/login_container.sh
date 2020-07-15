#!/usr/bin/env bash

# Change container name if the host machine is shared by multiple users.
CONTAINER="fuel"

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."
source ./tools/docker_version.sh

# Decide to use GPU or not.
DOCKER_RUN="docker run"
if [ -z "$(which nvidia-smi)" ]; then
  echo "No nvidia driver found"
elif [ -z "$(which nvidia-container-toolkit)" ]; then
  echo "No nvidia-container-toolkit found"
else
  DOCKER_RUN="docker run --gpus all --ipc=host"
fi

if [ -z "${HOME}" ]; then
  HOME=$( cd; pwd )
fi

docker ps -a --format "{{.Names}}" | grep -v gocloud | grep ${CONTAINER} > /dev/null
if [ $? -eq 0 ]; then
  echo "Found existing container. If you need a fresh one, run 'docker rm -f ${CONTAINER}' first."
else
  # Mount required volumes.
  FUEL_ROOT="$(pwd)"
  required_volumes="-v ${FUEL_ROOT}:/fuel"
  APOLLO_ROOT="$(cd ../apollo; pwd)"
  required_volumes="-v ${APOLLO_ROOT}:/apollo ${required_volumes}"

  # Populate user.bazelrc.
  mkdir -p local
  echo "startup --output_user_root=${FUEL_ROOT}/local/bazel_cache" > local/user.bazelrc
  required_volumes="-v ${FUEL_ROOT}/local:${FUEL_ROOT}/local ${required_volumes}"

  USER_ID=$(id -u)
  GRP=$(id -g -n)
  GRP_ID=$(id -g)

  # Mount optional volumes.
  optional_volumes=""

  APOLLO_MAP="../baidu/adu-lab/apollo-map"
  if [ -d "${APOLLO_MAP}" ]; then
    APOLLO_MAP="$(cd ${APOLLO_MAP}; pwd)"
    optional_volumes="-v ${APOLLO_MAP}:/apollo/modules/map/data ${optional_volumes}"
  fi

  CALIBRATION_DATA="../apollo-internal/modules_data/calibration/data"
  if [ -d "${CALIBRATION_DATA}" ]; then
    CALIBRATION_DATA="$(cd ${CALIBRATION_DATA}; pwd)"
    optional_volumes="-v ${CALIBRATION_DATA}:/apollo/modules/calibration/data ${optional_volumes}"
  fi
  
  DATA_VOLUME="/data"
  if [ -d "${DATA_VOLUME}" ]; then
    optional_volumes="-v ${DATA_VOLUME}:${DATA_VOLUME} ${optional_volumes}"
  fi

  SIMULATOR="../replay-engine"
  if [ -d "${SIMULATOR}" ]; then
    SIMULATOR="$(cd ${SIMULATOR}; pwd)"
    optional_volumes="-v ${SIMULATOR}:/apollo-simulator ${optional_volumes}"
  fi

  ${DOCKER_RUN} -it -d --privileged \
      --net host \
      --name ${CONTAINER} \
      --hostname ${CONTAINER} --add-host ${CONTAINER}:127.0.0.1 \
      ${required_volumes} \
      ${optional_volumes} \
      -w /fuel \
      -e DISPLAY=$DISPLAY \
      -e DOCKER_USER=$USER -e DOCKER_USER_ID=$USER_ID \
      -e DOCKER_GRP=$GRP -e DOCKER_GRP_ID=$GRP_ID \
      -e PYTHONPATH="/apollo/py_proto:/apollo/cyber/python:/apollo/bazel-bin/cyber/python/internal" \
      ${IMAGE} bash
  if [ "${USER}" != "root" ]; then
    docker exec ${CONTAINER} bash -c '/apollo/scripts/docker_start_user.sh'
  fi
  docker exec ${CONTAINER} bash -c "cat /home/libs/bash.rc >> ${HOME}/.bashrc"
fi
docker exec -it -u ${USER} ${CONTAINER} bash
