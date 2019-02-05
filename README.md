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
   # JDK 8
   sudo apt-get install openjdk-8-jdk
   sudo update-alternatives --config java

   # Conda env.
   /usr/local/miniconda2/bin/conda env update -f cluster/py27-conda.yaml
   source /usr/local/miniconda2/bin/activate py27
   ```

## Develop pipeline jobs

We leverage PySpark to orchestrate the jobs on Kubernetes cluster. Good
practices are:

1. Put all Python modules in ./fueling/ folder, and import them with full path
   like `import fueling.io.xxx`.

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
