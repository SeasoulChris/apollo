# Apollo Fuel

## Setup Env

1. Clone the repo along with other apollo repos:

   ```text
   - workspace
              | - apollo
              | - apollo-fuel      # Mounted as /apollo/modules/data/fuel
              | - apollo-internal
              | - apollo-prophet   # Mounted as /apollo/modules/data/prophet
              | - replay-engine
   ```

   `bstart`, `binto` and build Apollo as usual. Then you'll see the repo mounted
   at /apollo/modules/data/fuel, as a part of the whole apollo workspace.

1. [Install miniconda](https://docs.conda.io/en/latest/miniconda.html).
   If it's already installed, update it to the latest.
   (DO NOT RUN THIS IF YOU ARE USING PY27-CYBER)

   ```bash
   sudo /usr/local/miniconda/bin/conda update -n base -c defaults conda
   ```

1. Install env and activate.

   ```bash
   conda env update --prune -f fueling/conda/py27-cyber.yaml
   source activate fuel-py27-cyber
   ```

   Available envs are:
   * `fuel-py27-cyber` fueling/conda/py27-cyber.yaml
   * `fuel-py27` fueling/conda/py27.yaml
   * `fuel-py36` fueling/conda/py36.yaml

   Use the Cyber compatible env if you need to read, write Cyber records, or
   call Cyber functions. Otherwise, please use the standard envs.

### Ongoing efforts

1. `fuel-py27` is in deprecation along with Python 2.7 retiring. Any new jobs
   should NOT depend on it.
1. `fuel-py27-cyber` is in deprecation along with Cyber wrapper upgrade.
   * If the upcoming Python3 wrapper is compatible with the standard env， we'll
     unify everything into `fuel-py36`.
   * If the upcoming Python3 wrapper still has strong restrictions on lib
     versions, we'll maintain a new `fuel-py36-cyber`.

## Develop pipeline jobs

We leverage PySpark to orchestrate the jobs on Kubernetes cluster. Good
practices are:

1. Put all Python modules in ./fueling/ folder, and import them with full path
   like `import fueling.common.s3_utils`.
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
1. File and folder name convention: **lower_case_with_underscores** if it's
   importable, otherwise **lower-case-with-dash**. So all main pipeline jobs
   should be **dash** style.
1. Filter early, filter often.
1. Cascade simple transformations, instead of making a huge complicate one.
1. All transformations should be repeatable and consistant. The process and even
   the executor could fail any time, then the Spark will try to re-allocate the
   task to other peers. So be careful about letting flatMap() and
   flatMapValues() work with "yield" mappers. Because it's stateful, if a task
   failed unexpectedly, the pipeline have no idea about how to recover.
1. Reading record header is much faster than reading record, if you can do
   significatnt filtering on records according to its header, do it.
1. Use [gflags](https://abseil.io/docs/python/guides/flags), but don't abuse.
   Always run your pipeline with our tools, and put job file and gflags at end:
   `tools/submit-job-to-*.sh <tool-flags> <your-job.py> <job-gflags>`.

   To access flag values, you need to call `self.FLAGS[KEY]` from an instance of
   `BasePipeline`.
1. To learn more about PySpark APIs, please go to
   [Spark Docs](https://spark.apache.org/docs/latest/api/python/pyspark.html).

### Test your pipeline at local

```bash
tools/submit-job-to-local.sh /path/to/spark/job.py <gflags>
# Go to http://localhost:4040 when the server is launched successfully.
```

### Run pipeline in cluster

If you are pretty familliar with the infra, please:
1. Loop the data team in to have your job well reviewed and setup local k8s
   client.
1. Then run:

   ```bash
   tools/submit-job-to-k8s.sh \
       --env <python_env> \
       --worker <workers_count>  \
       --cpu <worker_cpu_count>  \
       --memory <worker_memory>  \
       --disk <worker_disk_GB>   \
       /path/to/spark/job.py <gflags>
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
