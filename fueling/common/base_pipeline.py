"""Fueling base pipeline."""
#!/usr/bin/env python
import os

import colored_glog as glog
import pyspark


class BasePipeline(object):
    """Fueling base pipeline."""
    SPARK_CONF = None

    def __init__(self, name):
        """Pipeline constructor."""
        self.name = name

    def context(self):
        """Get the SparkContext."""
        if self.SPARK_CONF is None:
            self.SPARK_CONF = pyspark.SparkConf().setAppName(self.name)
        return pyspark.SparkContext.getOrCreate(self.SPARK_CONF)

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
