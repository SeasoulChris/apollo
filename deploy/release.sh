#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

LOCAL_FUELING_PKG="deploy/fueling.zip"
REMOTE_FUELING_PKG="modules/data/jobs/deploy/$(date +%Y%m%d-%H%M%S)_fueling.zip"
rm -f "${LOCAL_FUELING_PKG}"

set -x
set -e

# Make new package
zip -r "${LOCAL_FUELING_PKG}" ./fueling -x *.pyc */__pycache__
# Upload new package
./apps/local/bos_fstool -s "${LOCAL_FUELING_PKG}" -d "${REMOTE_FUELING_PKG}"
# Update usage
sed -i "s|^FUELING=.*|FUELING=\"/mnt/bos/${REMOTE_FUELING_PKG}\"|g" \
    ./apps/lambda/bae-proxy/vehicle_calibration.sh
