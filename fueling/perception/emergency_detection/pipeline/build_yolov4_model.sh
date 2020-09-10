echo "Prepare building context"
BUILD_PATH="$(dirname "${BASH_SOURCE[0]}")"
BUILD_DIR=
CONTEXT="${BUILD_PATH}/${BUILD_DIR}"

echo "
FROM alpine
VOLUME /apollo/modules/perception/camera/lib/obstacle/detector/yolov4/
COPY ./yolov4_1_3_416_416.onnx /apollo/modules/perception/camera/lib/obstacle/detector/yolov4/
" > "${CONTEXT}"/Dockerfile

echo "Building auto labeler volume."
REPO="apolloauto/apollo"
TIMESTAMP_TAG="${REPO}:yolov4_volume-emergency_detection_model-$(date +%Y%m%d_%H%M)"
DEFAULT_TAG="${REPO}:yolov4_volume-emergency_detection_model-latest"
docker build -t ${TIMESTAMP_TAG} "${CONTEXT}" && \
docker tag ${TIMESTAMP_TAG} ${DEFAULT_TAG} && \

echo "Build auto labeler volume successfully!"
echo "To upload, please run"
printf "  docker push ${TIMESTAMP_TAG} && docker push ${DEFAULT_TAG}\n"
