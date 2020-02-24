#!/usr/bin/env bash

IMAGE="apolloauto/fuel-client:20200221_1712"
CONTAINER="fuel"

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Decide to use GPU or not.
DOCKER_RUN="docker run"
if [ -z "$(which nvidia-smi)" ]; then
  echo "No nvidia driver found"
elif [ -z "$(which nvidia-container-toolkit)" ]; then
  echo "No nvidia-container-toolkit found"
else
  DOCKER_RUN="docker run --gpus all"
fi

docker ps -a --format "{{.Names}}" | grep ${CONTAINER} > /dev/null
if [ $? -eq 0 ]; then
  echo "Found existing container. If you need a fresh one, run 'docker rm -f ${CONTAINER}' first."
else
  required_volumes="-v $(pwd):/fuel"

  APOLLO_ROOT="$(cd ../apollo-bazel2.x; pwd)"
  required_volumes="-v ${APOLLO_ROOT}:/apollo ${required_volumes}"

  USER_ID=$(id -u)
  GRP=$(id -g -n)
  GRP_ID=$(id -g)

  CACHE_DIR="/home/.bazel_cache"
  sudo mkdir -p ${CACHE_DIR}
  sudo chown ${GRP}:${USER} ${CACHE_DIR}
  required_volumes="-v ${CACHE_DIR}:${CACHE_DIR} ${required_volumes}"

  optional_volumes=""
  APOLLO_MAP="../baidu/adu-lab/apollo-map"
  if [ -d "${APOLLO_MAP}" ]; then
    APOLLO_MAP="$(cd ${APOLLO_MAP}; pwd)"
    optional_volumes="-v ${APOLLO_MAP}:/apollo/modules/map/data ${optional_volumes}"
  fi

  ${DOCKER_RUN} -it -d --privileged \
      --net host \
      --name ${CONTAINER} \
      --hostname ${CONTAINER} --add-host ${CONTAINER}:127.0.0.1 \
      ${required_volumes} \
      ${optional_volumes} \
      -w /fuel \
      -e DOCKER_USER=$USER -e DOCKER_USER_ID=$USER_ID \
      -e DOCKER_GRP=$GRP -e DOCKER_GRP_ID=$GRP_ID \
      ${IMAGE} bash
  if [ "${USER}" != "root" ]; then
    docker exec ${CONTAINER} bash -c '/apollo/scripts/docker_adduser.sh'
    HOME="/home/${USER}"
  else
    HOME="/root"
  fi
  docker exec ${CONTAINER} bash -c "cat /home/libs/bash.rc >> ${HOME}/.bashrc"
fi

docker exec -it -u ${USER} ${CONTAINER} bash
