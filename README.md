# Apollo Fuel

## Setup Env

1. Find a proper folder to clone the following two repos:

   ```bash
   git clone --single-branch --branch bazel2.x git@github.com:ApolloAuto/apollo.git apollo-bazel2.x
   git clone git@github.com:<YourAccount>/apollo-fuel.git
   ```

   So that you have a workspace which looks like:

   - apollo-bazel2.x
   - apollo-fuel

1. Then go to the apollo-fuel repo, start a container.

   ```bash
   cd apollo-fuel
   ./tools/login_container.sh
   ```

   Now you should be in `/fuel` which maps to apollo-fuel, and there is also `/apollo` which maps to
   apollo-bazel2.x.

1. Please note that if you run `login_container.sh` again you are entering the same container. In
   case a fresh container is needed, please run the following command first.

   ```bash
   docker rm -f fuel
   ```

## Build and Run

Everything is managed by pure [Bazel](https://docs.bazel.build/versions/master/be/python.html).
Generally you need a BUILD target for each python file, which could be one of

* `py_library(name="lib_target", ...)`
* `py_binary(name="bin_target", ...)`
* `py_test(name="test_target", ...)`

1. Simply build everything:

   ```bash
   ./tools/build.sh
   ```

   *If keep encountering pip-fetching-timeout issue, you can try building with pre-fetched cache*

   ```bash
   ./tools/build_local.sh
   ```

1. Goto [fueling/demo](https://github.com/ApolloAuto/apollo-fuel/tree/master/fueling/demo)
   to check our demos.

1. To build any target:

   ```bash
   bazel build //path/to:target
   ```

1. To run a binary target:

   ```bash
   # Run at local.
   bazel run //path/to:target
   # Run with some flags.
   bazel run //path/to:target -- <flags>
   # Get help information.
   bazel run //path/to:target -- --help
   # Get help information and all available flags.
   bazel run //path/to:target -- --helpfull
   # Run a pipeline in cloud.
   bazel run //path/to:target -- --cloud <flags>
   ```

1. To run a unit test target:

   ```bash
   bazel test //path/to:test_target
   ```

## Develop pipeline jobs

We leverage PySpark to orchestrate the jobs on Kubernetes cluster. Good practices are:

1. Put all Python modules in ./fueling/ folder, and import them with full path like
   `import fueling.common.file_utils`.
1. Inherit the `fueling.common.base_pipeline.BasePipeline` and implement your own `run()` function.
   Make sure that `bazel build //path/to/your:target` passes.
1. Inherit the `fueling.common.base_pipeline.BasePipelineTest` and implement your own `test_xxx`
   functions. Make sure that `bazel test //path/to/your:target_test` passes.
1. Comment intensively and accurately. Every RDD should be well described with the pattern.

   ```python
   # RDD(element_type), other comments if any.
   # PairRDD(key_type, value_type), other comments if any.
   ```

1. Import only what you need in each file. And split them into sections in order:

   ```python
   import standard_packages

   import third_party_packages

   import apollo_packages

   import fueling_packages
   ```

   * It's OK to use `from package import Type` and `import package as alias` statements.
   * Each section should be in alphabetical order.

1. Python script and package name convention: **lower_case_with_underscores**.
1. Filter early, filter often.
1. Cascade simple transformations, instead of making a huge complicate one.
1. All transformations should be repeatable and consistant. The process and even the executor could
   fail any time, then the Spark will try to re-allocate the task to other peers. So be careful
   about letting `flatMap()` and `flatMapValues()` work with "yield" mappers. Because it's stateful,
   if a task failed unexpectedly, the pipeline have no idea about how to recover.
1. Reading record header is much faster than reading record, if you can do significant filtering on
   records according to its header, do it.
1. Use `absl.flags` for script argument. refer to
   [Flags And Logging guide](docs/flags-and-logging-guide.md) for more information.
1. Refer to files:

   1) Use bazel [filegroup](https://docs.bazel.build/versions/master/be/general.html#filegroup) to
      bind data files which are hosted in Fuel. Find example usage in fueling/demo.
   1) Use absolute path like `/apollo/...` to refer to files in Apollo.
   1) Goto next tip for cloud data usage.

1. We use [bosfs](https://cloud.baidu.com/doc/BOS/s/Ajwvyqhya) to mount Apollo's BOS storage at
   `/mnt/bos`, which can be used as a POSIX file system. But if you want to list many files under a
   folder, a better way is to call
   [`a_BasePipeline_instance.our_storage().list_files(...)`](fueling/common/bos_client.py#L74) which
   leverages [boto3](https://cloud.baidu.com/doc/BOS/s/ojwvyq973#aws-sdk-for-python) to do efficient
   S3-style queries. Please read these documents carefully, which will improve your pipeline a lot.
1. To learn more about PySpark APIs, please go to
   [Spark Docs](https://spark.apache.org/docs/latest/api/python/pyspark.html).

### Debug

1. Goto [usa-data.baidu.com](http://usa-data.baidu.com:8001/api/v1/namespaces/default/services/http:warehouse-service:8000/proxy/jobs)
   to find your job, and explore around to find the pods and logs.

1. Please request resources carefully, as it may block other teammates' work.

   * Increase workers if you need better parallelism.
   * Increase CPU if your job is computing intensive.
   * Increase memory if you saw out-of-memory error.
   * Increase disk if you saw running out of ephemeral stroage.

Note that all your actions and jobs are under monitoring.
