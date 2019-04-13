"""Fueling base pipeline."""
#!/usr/bin/env python
import os

from pyspark import SparkConf, SparkContext
import colored_glog as glog


class BasePipeline(object):
    """Fueling base pipeline."""
    SPARK_CONTEXT = None

    def __init__(self, name):
        """Pipeline constructor."""
        self.name = name
        BasePipeline.SPARK_CONTEXT = SparkContext.getOrCreate(SparkConf().setAppName(self.name))

    @classmethod
    def context(cls):
        """Get the SparkContext."""
        if cls.SPARK_CONTEXT is None:
            cls.SPARK_CONTEXT = SparkContext.getOrCreate(SparkConf().setAppName('BasePipeline'))
        return cls.SPARK_CONTEXT

    def run_test(self):
        """Run the pipeline in test mode."""
        raise Exception('{}::run_test not implemented!'.format(self.name))

    def run_prod(self):
        """Run the pipeline in production mode."""
        raise Exception('{}::run_prod not implemented!'.format(self.name))

    def run_grpc(self):
        """Run the pipeline in GRPC mode."""
        raise Exception('{}::run_grpc not implemented!'.format(self.name))

    def main(self):
        """Run the pipeline."""
        RUNNING_MODE = os.environ.get('APOLLO_FUEL_MODE', 'PROD')
        glog.info('Running {} pipeline in {} mode'.format(self.name, RUNNING_MODE))
        if RUNNING_MODE == 'TEST':
            self.run_test()
        elif RUNNING_MODE == 'GRPC':
            self.run_grpc()
        else:
            self.run_prod()
