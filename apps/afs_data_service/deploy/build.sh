#!/usr/bin/env bash

DOCKER_REGISTRY="hub.baidubce.com"
DOCKER_USER="apollofuel"

REPO="${DOCKER_REGISTRY}/${DOCKER_USER}/afs-data-service"
IMAGE="${REPO}:$(date +%Y%m%d_%H%M)"

echo "Building image: ${IMAGE}"

cd "$( dirname "${BASH_SOURCE[0]}" )"
set -ex

# Prepare source
SRC_DIR="server"
ADBSDK_DIR="adbsdk"
mkdir -p ${SRC_DIR}
cp ../server.py ${SRC_DIR}
cp ../proto/* ${SRC_DIR}
if [ ! -d "./${ADBSDK_DIR}" ]; then
  # wiki: http://wiki.baidu.com/pages/viewpage.action?pageId=1040599879
  # http://agile.baidu.com/#/release/baidu/adu/adbsdk
  wget -O output.tar.gz --no-check-certificate \
      --header "IREPO-TOKEN:89938b26-fc83-4d84-bfdb-a464a66613bd" \
      "https://irepo.baidu-int.com/rest/prod/v3/baidu/adu/adbsdk/releases/1.0.3.1/files" 
  mkdir adbsdk_tmp
  tar zxf output.tar.gz -C adbsdk_tmp
  tar zxf adbsdk_tmp/output/adbsdk*.tar.gz
  rm -r adbsdk_tmp
  rm output.tar.gz
fi

# Build.
docker build -t ${IMAGE} --network host .
rm -rf ${SRC_DIR}
docker push ${IMAGE}

# Deploy.
cp ~/.kube/config kube.config.original
sudo cp kube.config ~/.kube/config
sed -i "s|image: ${REPO}.*|image: ${IMAGE}|g" deploy.yaml
kubectl apply -f deploy.yaml --validate=false
sudo mv kube.config.original ~/.kube/config
