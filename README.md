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

   `bstart`, `binto` and build Apollo as usual.

1. Install additional tools for the Apollo container.

   ```bash
   # Upgrade conda first.
   sudo /usr/local/miniconda2/bin/conda update -n base -c defaults conda
   # Install and activate env. Currently we only have Python 2.7, because
   # Python 3.6 depends on Cyber python wrapper upgrade.
   sudo rm -fr /usr/local/miniconda2/envs/py27
   conda env update -f cluster/py27-conda.yaml
   source activate py27
   ```

## Develop pipeline jobs

We leverage PySpark to orchestrate the jobs on Kubernetes cluster. Good
practices are:

1. Put all Python modules in ./fueling/ folder, and import them with full path
   like `import fueling.common.s3_utils`.

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
