#!/usr/bin/env bash

IMAGE="apolloauto/apollo:dev-18.04-x86_64-20200130_1429"
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
  APOLLO_ROOT="$(cd ../apollo; pwd)"
  USER_ID=$(id -u)
  GRP=$(id -g -n)
  GRP_ID=$(id -g)
  ${DOCKER_RUN} -it -d --privileged \
      --net host \
      --name ${CONTAINER} --hostname ${CONTAINER} \
      -v "$(pwd):/fuel" \
      -v "${APOLLO_ROOT}:/apollo" \
      -w /fuel \
      -e DOCKER_USER=$USER \
      -e DOCKER_USER_ID=$USER_ID \
      -e DOCKER_GRP="$GRP" \
      -e DOCKER_GRP_ID=$GRP_ID \
      ${IMAGE} bash
  if [ "${USER}" != "root" ]; then
    docker exec ${CONTAINER} bash -c '/apollo/scripts/docker_adduser.sh'
  fi
fi

docker exec -it -u ${USER} ${CONTAINER} bash
