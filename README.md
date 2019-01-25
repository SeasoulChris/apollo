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

## Develop pipeline jobs

We leverage PySpark to orchestrate the jobs on Kubernetes cluster. Good
practices are:

1. Put all Python modules in ./fueling/ folder, and import them with full path
   like `import fueling.io.xxx`.
