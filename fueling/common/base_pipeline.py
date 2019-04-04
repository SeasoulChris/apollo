"""Fueling base pipeline."""
#!/usr/bin/env python
import os

import pyspark_utils.helper as spark_helper

import fueling.common.colored_glog as glog


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

    def main(self):
        """Run the pipeline."""
        if os.environ.get('APOLLO_FUEL_MODE') == 'TEST':
            glog.info('Running {} pipeline in TEST mode'.format(self.name))
            self.run_test()
        else:
            glog.info('Running {} pipeline in PROD mode'.format(self.name))
            self.run_prod()
