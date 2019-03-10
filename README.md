# Apollo Fuel

## Setup Env

1. Clone the repo along with other apollo repos:

   ```text
   - workspace
              | - apollo
              | - apollo-fuel
              | - apollo-internal
              | - replay-engine
   ```

   `bstart`, `binto` and build Apollo as usual. Then you'll see the repo mounted
   at /apollo/modules/data/fuel, as a part of the whole apollo workspace.

1. [Install miniconda](https://docs.conda.io/en/latest/miniconda.html).

   If it's already installed, update it to the latest.

   ```bash
   sudo /usr/local/miniconda2/bin/conda update -n base -c defaults conda
   ```

1. Install env and activate.

   ```bash
   conda env update -f configs/conda-py27-cyber.yaml
   source activate fuel-py27-cyber
   ```

   Available envs are:
   * `fuel-py27-cyber` configs/conda-py27-cyber.yaml
   * `fuel-py27` configs/conda-py27.yaml
   * `fuel-py36` configs/conda-py36.yaml

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

And good coding practices are:

1. Filter early, filter often.

1. Cascade simple transformations, instead of making a huge complicate one.

1. All transformations should be repeatable and consistant. The process and even
   the executor could fail any time, then the Spark will try to re-allocate the
   task to other peers. So be careful about letting flatMap() and
   flatMapValues() work with "yield" mappers. Because it's stateful, if a task
   failed unexpectedly, the pipeline have no idea about how to recover.

1. Reading record header is much faster than reading record, if you can do
   significatnt filtering on records according to its header, do it.

### Test your pipeline at local

```bash
tools/submit-job-to-local.sh /path/to/spark/job.py
# Go to http://localhost:4040 when the server is launched successfully.
```

### Run pipeline in cluster

Check in your code to add the job to a pipeline carrier, which might run once,
daily or weekly according to your need.

Talk to the data team (usa-data@baidu.com) if you are pretty familliar with the
infra and want to get more control.

## TODO

1. Support private docker repo. This should be doable by add k8s secret:

   ```bash
   kubectl create secret docker-registry "<secret_name>" \
       --docker-server="docker.io" \
       --docker-username="${DOCKER_USER}" \
       --docker-password="${DOCKER_PASSWORD}" \
       --docker-email="xxx@baidu.com"
   ```

   and then reference it with

   ```bash
   spark-submit ... --conf spark.kubernetes.container.image.pullSecrets="<secret_name>"
   ```

   But in a quick trial it failed for the Baidu CCE. Not sure if it's a bug for
   us or for the cloud provider.
