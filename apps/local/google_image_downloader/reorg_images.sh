#!/usr/bin/env bash

INPUT_DIR=$1
OUTPUT_DIR=$2

set -x

find ${INPUT_DIR} -type f | while read -r IMAGE; do
  IMAGE_DIR=$(dirname "$IMAGE")
  NEW_DIR=${OUTPUT_DIR}/$(echo "${IMAGE_DIR}" | sha1sum | awk '{print $1}')
  mkdir -p "${NEW_DIR}"
  echo "$(basename "${IMAGE_DIR}")" > "${NEW_DIR}/keyword.txt"

  EXTENSION="${IMAGE##*.}"
  NEW_IMAGE=${NEW_DIR}/$(echo "${IMAGE}" | sha1sum | awk '{print $1}').${EXTENSION}
  echo "Moving ${IMAGE} to ${NEW_IMAGE}"
  mv "${IMAGE}" "${NEW_IMAGE}"
done
