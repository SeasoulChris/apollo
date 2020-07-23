#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )/../.."

source ./apps/autotuner/common.sh
source ./tools/docker_version.sh

function print_usage() {
    echo "Usage:
    ./build.sh [ $(join_by ' | ' ${VALID_CLUSTERS[@]}) ]
    "
}

function build_and_push() {
  set -ex
  local github_version=$(git log --pretty=format:'%H (%cd)' -n 1)
  local dockerfile="apps/autotuner/docker/Dockerfile"
  sed -i "s|___FUEL_DOCKER_VERSION__|${IMAGE}|g" $dockerfile
  docker build -t ${IMAGE_NAME} --network host --build-arg GITHUB_VERSION="${github_version}" -f $dockerfile .
  git checkout -- $dockerfile

  echo 'Start pushing cost_service image ...'
  TAG="$(date +%Y%m%d_%H%M)"
  docker tag ${IMAGE_NAME} "${DEST_REPO}/${IMAGE_NAME}:${TAG}"
  docker push "${DEST_REPO}/${IMAGE_NAME}:${TAG}"

  TAG="latest"
  docker tag ${IMAGE_NAME} "${DEST_REPO}/${IMAGE_NAME}:${TAG}"
  docker push "${DEST_REPO}/${IMAGE_NAME}:${TAG}"
}

function main() {
  if [ $# -ne 1 ]; then
    print_usage
    exit 1
  fi

  CLUSTER=$1
  check_cluster $CLUSTER
  init_settings
  build_and_push
}

main "$@"
