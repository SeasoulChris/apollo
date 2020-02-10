# Apollo Fuel 2.0

## Setup Env

1. Find a proper folder to clone the following two repos:

   ```bash
   git clone --single-branch --branch bazel2.x git@github.com:<YourAccount>/apollo.git apollo-bazel2.x
   git clone git@github.com:<YourAccount>/apollo-fuel.git
   ```

1. Then go to the fuel repo, start the container, build everything.

   ```bash
   cd apollo-fuel
   ./v2/login_container.sh
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

1. You can run `bazel build //path/to:target` to build any target.
1. You can run `bazel run //path/to:bin_target` to run a binary target.
1. You can run `bazel test //path/to:test_target` to run a unit test target.

Every `py_binary` target will be also be built into an executable zip file. You can run it in cloud
with `./v2/cloud_run.py bazel-bin/path/to/bin_target.zip`. Feel free to run some live demos located
at `fueling/demo`.
