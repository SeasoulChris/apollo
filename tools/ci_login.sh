#!/usr/bin/env bash

# Change container name if the host machine is shared by multiple users.
CONTAINER="fuel"

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."
source ./tools/docker_version.sh

DOCKER_RUN="docker run"

if [ -z "${HOME}" ]; then
  HOME=$( cd; pwd )
fi

docker ps -a --format "{{.Names}}" | grep ${CONTAINER} > /dev/null
if [ $? -eq 0 ]; then
  echo "Found existing container. If you need a fresh one, run 'docker rm -f ${CONTAINER}' first."
else
  # Mount required volumes.
  required_volumes="-v $(pwd):/fuel"

  APOLLO_ROOT="/home/apollo/apollo"
  if [ -d ${APOLLO_ROOT} ]; then
    cd ${APOLLO_ROOT}
    git pull origin master
  else
    git clone --depth 1 git@github.com:ApolloAuto/apollo.git ${APOLLO_ROOT}
  fi
  required_volumes="-v ${APOLLO_ROOT}:/apollo ${required_volumes}"

  USER_ID=$(id -u)
  GRP=$(id -g -n)
  GRP_ID=$(id -g)

  # To support multi-containers on shared host.
  LOCAL_CACHE_DIR="${HOME}/.cache/bazel/${CONTAINER}"
  CONTAINER_CACHE_DIR="/home/.bazel_cache"
  mkdir -p ${LOCAL_CACHE_DIR}
  required_volumes="-v ${LOCAL_CACHE_DIR}:${CONTAINER_CACHE_DIR} ${required_volumes}"

  ${DOCKER_RUN} -it -d --privileged \
      --net host \
      --name ${CONTAINER} \
      ${required_volumes} \
      -w /fuel \
      -e DOCKER_USER=$USER -e DOCKER_USER_ID=$USER_ID \
      -e DOCKER_GRP=$GRP -e DOCKER_GRP_ID=$GRP_ID \
      ${IMAGE} bash
  if [ "${USER}" != "root" ]; then
    docker exec ${CONTAINER} bash -c '/apollo/scripts/docker_adduser.sh'
  fi
fi
