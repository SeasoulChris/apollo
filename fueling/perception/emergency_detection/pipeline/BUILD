load("@rules_python//python:defs.bzl", "py_binary")
package(default_visibility = ["//visibility:public"])

py_binary(
	name = "auto_label_pipeline",
	srcs = ["auto_label_pipeline.py"],
	deps = [
		"@yolov4",
		"//fueling/common:base_pipeline",
	],
)