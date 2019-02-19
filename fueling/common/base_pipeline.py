"""Fueling base pipeline."""
#!/usr/bin/env python

import pyspark_utils.helper as spark_helper


class BasePipeline(object):
    """Fueling base pipeline."""

    def __init__(self, name):
        """Pipeline constructor."""
        self.name = name

    def get_spark_context(self):
        """Get the SparkContext."""
        return spark_helper.get_context(self.name)

    def run_test(self):
        """Run the pipeline in test mode."""
        raise Exception('{}::RunTest not implemented!'.format(self.name))

    def run_prod(self):
        """Run the pipeline in production mode."""
        raise Exception('{}::RunProd not implemented!'.format(self.name))
