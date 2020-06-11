#!/bin/bash

source ./common.sh

function print_usage() {
    echo "Usage:
    ./build.sh [ $(join_by ' | ' ${VALID_CLUSTERS[@]}) ]
    "
}

function build_and_push() {
  GITHUB_VERSION=$(git log --pretty=format:'%H (%cd)' -n 1)
  echo "Building cost_service image w/ github version $GITHUB_VERSION ..."
  cd $( dirname "${BASH_SOURCE[0]}" )/../..

  set -ex
  docker build -t ${IMAGE} --network host --build-arg GITHUB_VERSION="$GITHUB_VERSION"  -f apps/autotuner/docker/Dockerfile .

  echo 'Start pushing cost_service image ...'
  TAG="$(date +%Y%m%d_%H%M)"
  docker tag ${IMAGE} "${DEST_REPO}/${IMAGE}:${TAG}"
  docker push "${DEST_REPO}/${IMAGE}:${TAG}"

  TAG="latest"
  docker tag ${IMAGE} "${DEST_REPO}/${IMAGE}:${TAG}"
  docker push "${DEST_REPO}/${IMAGE}:${TAG}"
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
