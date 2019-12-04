# Apollo Fuel

## Setup Env

1. Clone the repo along with other apollo repos:

   ```text
   - workspace
              | - apollo
              | - apollo-fuel      # Mounted as /apollo/modules/data/fuel
              | - apollo-internal
   ```

   `bstart`, `binto` and build Apollo as usual. Then you'll see the repo mounted
   at /apollo/modules/data/fuel, as a part of the whole apollo workspace.

1. Install env and activate.

   ```bash
   conda env update --prune -f conda/py36.yaml
   source activate fuel-py36
   ```

## Develop pipeline jobs

We leverage PySpark to orchestrate the jobs on Kubernetes cluster. Good
practices are:

1. Put all Python modules in ./fueling/ folder, and import them with full path
   like `import fueling.common.file_utils`.
1. Inherit the `fueling.common.base_pipeline.BasePipeline` and implement your
   own `run_test()` and `run_prod()` functions. Generally they should share most
   procedures and only differ in input and output locations or scale.
1. Put all test data in ./testdata/module/... folder, make sure your job works
   perfectly at local. And when it is submited to a cluster, the huge test files
   are efficiently ignored.
1. Comment intensively and accurately. Every RDD should be well described with
   the pattern.

   ```python
   # RDD(element_type), other comments if any.
   # PairRDD(key_type, value_type), other comments if any.
   ```

1. Import only what you need in each file. And split them into sections in
   order:

   ```python
   import standard_packages

   import third_party_packages

   import apollo_packages

   import fueling_packages
   ```

   * It's OK to use `from package import Type` and `import package as alias`
     statements.
   * Each section should be in alphabetical order.

1. Always try to use **absolute file paths**, as apollo-fuel integrates multiple
   upstream environments which have their own well-known working directories,
   such as `/apollo`, `/mnt/bos`, `/opt/spark/work-dir`, etc. To avoid relative
   path mistakes, we enforce providing absolute file paths whenever possible.
1. Python script and package name convention: **lower_case_with_underscores**.
1. Filter early, filter often.
1. Cascade simple transformations, instead of making a huge complicate one.
1. All transformations should be repeatable and consistant. The process and even
   the executor could fail any time, then the Spark will try to re-allocate the
   task to other peers. So be careful about letting flatMap() and
   flatMapValues() work with "yield" mappers. Because it's stateful, if a task
   failed unexpectedly, the pipeline have no idea about how to recover.
1. Reading record header is much faster than reading record, if you can do
   significatnt filtering on records according to its header, do it.
1. Use `absl.flags` for script argument. refer to
   [Flags And Logging guide](docs/flags-and-logging-guide.md) for more
   information.
1. For Apollo data access, we use [bosfs](https://cloud.baidu.com/doc/BOS/s/Ajwvyqhya)
   to mount Apollo's BOS storage at `/mnt/bos`, which can be used as a POSIX
   file system. But if you want to list many files under a folder, a better way
   is to call [`a_BasePipeline_instance.our_storage().list_files(...)`](fueling/common/bos_client.py#L74)
   which leverages [boto3](https://cloud.baidu.com/doc/BOS/s/ojwvyq973#aws-sdk-for-python)
   to do efficient S3-style queries. Please read these documents carefully,
   which will improve your pipeline a lot.
1. To learn more about PySpark APIs, please go to
   [Spark Docs](https://spark.apache.org/docs/latest/api/python/pyspark.html).

### Test your pipeline at local

```bash
# Get to know the options.
./tools/submit-job-to-local.sh -h

# Start a local job.
./tools/submit-job-to-local.sh [options] /path/to/spark/job.py [job-gflags]
# Go to http://localhost:4040 when the server is launched successfully.
```

As the environment changes and libraries upgrade frequently, you'd better mount
your job to the [regression test train](deploy/regression_test.sh), which we
keep a close eye on, and make sure it passes before pushing a new docker image,
so as to minimize the chance to surprise you.

### Run pipeline in cluster

If you are pretty familliar with the infra, please:
1. Loop the data team in to have your job well reviewed.

1. Then run:

   ```bash
   # Get to know the options.
   ./tools/submit-job-to-k8s.py --help

   # Start a cloud job.
   ./tools/submit-job-to-k8s.py --main=/path/to/spark/job.py [other options]

   # Find your job and access its Spark UI.
   ./tools/access-service-on-k8s.sh 4040
   ```

1. Monitor jobs with general `kubectl` commands.

   * `kubectl get pods`, you'll find running PODs in the cluster.
   * `kubectl logs -f <POD>`, you'll see all logs on that POD. Generally you may
     need to check the `driver` logs often, and sometimes check the `exec` logs
     to debug the executor code.
   * Please study this
     [Cheet Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet) to
     get more from the cluster!

1. Please request resources carefully, as it may block other teammates' work.

   * Increase workers if you need better parallelism.
   * Increase CPU if your job is computing intensive.
   * Increase memory if you saw out-of-memory error.
   * Increase disk if you saw running out of ephemeral stroage.

Note that all your actions and jobs are under monitoring.
