# Apollo Fuel 2.0

## Setup Env

1. Find a proper folder to clone the following two repos:

   ```bash
   git clone --single-branch --branch bazel2.x git@github.com:<YourAccount>/apollo.git apollo-bazel2.x
   git clone git@github.com:<YourAccount>/apollo-fuel.git
   ```

   So that you have a workspace which looks like:

   - apollo-bazel2.x
   - apollo-fuel

1. Then go to the apollo-fuel repo, start a container, build everything.

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
