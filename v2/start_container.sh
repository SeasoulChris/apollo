#!/usr/bin/env bash

IMAGE="local:fuel"
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
      --name ${CONTAINER} \
      --hostname ${CONTAINER} --add-host ${CONTAINER}:127.0.0.1 \
      -v "$(pwd):/fuel" \
      -v "${APOLLO_ROOT}:/apollo" \
      -v "/tmp:/tmp" \
      -w /fuel \
      -e DOCKER_USER=$USER -e DOCKER_USER_ID=$USER_ID \
      -e DOCKER_GRP=$GRP -e DOCKER_GRP_ID=$GRP_ID \
      ${IMAGE} bash
  if [ "${USER}" != "root" ]; then
    docker exec ${CONTAINER} bash -c '/apollo/scripts/docker_adduser.sh'
    docker exec ${CONTAINER} bash -c "echo 'source /usr/local/miniconda/bin/activate fuel' >> /home/${USER}/.bashrc"
    docker exec ${CONTAINER} bash -c "echo 'source /usr/local/lib/bazel/bin/bazel-complete.bash' >> /home/${USER}/.bashrc"
  else
    docker exec ${CONTAINER} bash -c "echo 'source /usr/local/miniconda/bin/activate fuel' >> /${USER}/.bashrc"
    docker exec ${CONTAINER} bash -c "echo 'source /usr/local/lib/bazel/bin/bazel-complete.bash' >> /${USER}/.bashrc"
  fi
fi

docker exec -it -u ${USER} ${CONTAINER} bash
