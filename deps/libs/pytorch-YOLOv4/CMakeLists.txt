
cmake_minimum_required(VERSION 2.8.11)
#Project
project(YOLOv4)

find_package( OpenCV )

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/xxx/TensorRT-7.1.3.4/include")
include_directories("/usr/local/cuda-10.2/include")
		

#Output
add_executable(
	yolov4_tensorrt_test
	yolov4_obstacle_detector_test.cc
)

target_link_libraries(yolov4_tensorrt_test ${OpenCV_LIBS} 
	/xxx/TensorRT-7.1.3.4/lib/libnvinfer.so
	/xxx/TensorRT-7.1.3.4/lib/libnvinfer_plugin.so
	/xxx/TensorRT-7.1.3.4/lib/libnvonnxparser.so
	/xxx/TensorRT-7.1.3.4/lib/libnvparsers.so
	/xxx/TensorRT-7.1.3.4/lib/libmyelin.so
	/xxx/TensorRT-7.1.3.4/lib/libnvcaffe_parser.so
	/usr/local/cuda-10.2/lib64/libcudart.so
	)