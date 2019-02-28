# Apollo Fuel

## Setup

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

1. Install additional tools for the Apollo container.

   ```bash
   # Upgrade conda first.
   sudo /usr/local/miniconda2/bin/conda update -n base -c defaults conda
   # Install and activate env. Currently we only have Python 2.7, because
   # Python 3.6 depends on Cyber python wrapper upgrade.
   sudo rm -fr /usr/local/miniconda2/envs/py27
   conda env update -f configs/py27-conda.yaml
   source activate py27
   ```

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
