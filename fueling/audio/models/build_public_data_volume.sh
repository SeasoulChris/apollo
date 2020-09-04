echo "Prepare building context"
BUILD_PATH="$(dirname "${BASH_SOURCE[0]}")"
BUILD_DIR=
CONTEXT="${BUILD_PATH}/${BUILD_DIR}"

echo "
FROM alpine
VOLUME /apollo/modules/audio/data/
COPY ./torch_siren_detection_model.pt /apollo/modules/audio/data/
" > "${CONTEXT}"/Dockerfile


echo "Building data volume."
REPO="apolloauto/apollo"
TIMESTAMP_TAG="${REPO}:data_volume-audio_model-$(date +%Y%m%d_%H%M)"
DEFAULT_TAG="${REPO}:data_volume-audio_model-latest"
docker build -t ${TIMESTAMP_TAG} "${CONTEXT}" && \
docker tag ${TIMESTAMP_TAG} ${DEFAULT_TAG} && \

echo "Build data volume successfully!"
echo "To upload, please run"
printf "  docker push ${TIMESTAMP_TAG} && docker push ${DEFAULT_TAG}\n"
