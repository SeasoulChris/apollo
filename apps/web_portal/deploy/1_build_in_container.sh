#!/usr/bin/env bash

TIME=$(date +%Y%m%d-%H%M)
PACKAGE="web_portal_${TIME}.zip"

bazel build //apps/web_portal:index
zip -j "${PACKAGE}" /fuel/bazel-bin/apps/web_portal/index.zip
echo "Packaged ${PACKAGE}"
