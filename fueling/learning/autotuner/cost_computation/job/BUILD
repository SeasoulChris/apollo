package(default_visibility = ["//visibility:public"])

py_library(
    name = "base_cost_job",
    srcs = ["base_cost_job.py"],
    deps = [
        "//fueling/common:logging",
    ]
)

py_library(
    name = "local_cost_job",
    srcs = ["local_cost_job.py"],
    deps = [
        ":base_cost_job",
    ]
)

py_library(
    name = "k8s_cost_job",
    srcs = ["k8s_cost_job.py"],
    deps = [
        ":base_cost_job",
        "//apps/k8s/spark_submitter:client",
        "//fueling/common:logging",
        "//fueling/learning/autotuner/common:utils",
    ]
)
