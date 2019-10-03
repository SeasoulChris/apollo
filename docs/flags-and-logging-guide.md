# Flags and Logging Guide

We leverage *Google Abseil* as fundamental infra, which includes utilities like
flags, logging, etc. It's also the successor of some well-known utils like
gflags and glog.

Please check the official docs for basic usage:

* [Quick Start Guide](https://abseil.io/docs/python/quickstart)
* [Flags](https://abseil.io/docs/python/guides/flags)
* [Logging](https://abseil.io/docs/python/guides/logging)

## What are different in PySpark pipeline?

### Flags

As you know, PySpark operations are distributed across the cluster, while FLAG
values are not. Fortunately, we will do the distribution for you.

Instead of using `flags.FLAGS.my_flag`, we stored all flag values into
`BasePipeline.FLAGS` dict, and distribute them to all Spark driver and
executors. So you are safe to use it anywhere in your pipeline.

```python
class IndexRecords(BasePipeline):
    def run_prod(self):
        print(self.FLAGS['my_flag'])
```

### Logging

`absl.logging` needs necessary initialization, while Spark executors don't have
a chance to do that. So we wrapped it to `fueling.common.logging` which
guarantees one-time setup. All you need to do is

```python
import fueling.common.logging as logging
```

Then use it just like `absl.logging`.
