"""Fueling base pipeline."""
#!/usr/bin/env python
import sys

from absl import app, flags
from pyspark import SparkConf, SparkContext
import colored_glog as glog


flags.DEFINE_string('running_mode', None, 'Pipeline running mode: TEST, PROD or GRPC.')
flags.DEFINE_integer('executors', 0, 'Pipeline executor count. 0 means unknown')


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

    def __main__(self, argv):
        """"""
        mode = flags.FLAGS.running_mode
        if mode is None:
            glog.fatal('No running mode is specified! Please run the pipeline with /tools/<runner> '
                       'instead of calling native "python".')
            sys.exit(1)

        glog.info('Running {} pipeline in {} mode'.format(self.name, mode))
        if mode == 'TEST':
            self.run_test()
        elif mode == 'GRPC':
            self.run_grpc()
        else:
            self.run_prod()

    def main(self):
        """Run the pipeline."""
        app.run(self.__main__)
