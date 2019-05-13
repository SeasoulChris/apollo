#!/usr/bin/env bash
SRC_DIR=$1
TARGET_DIR=$2

set -e

/apollo-simulator/bazel-bin/sim_control/dynamic_model/echo_lincoln_pipeline/echo_lincoln_bin \
    < ${SRC_DIR} > ${TARGET_DIR}
