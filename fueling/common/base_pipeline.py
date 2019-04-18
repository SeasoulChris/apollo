"""Fueling base pipeline."""
#!/usr/bin/env python
import sys

from absl import app, flags
from pyspark import SparkConf, SparkContext
import colored_glog as glog

from fueling.common.mongo_utils import Mongo


flags.DEFINE_string('running_mode', None, 'Pipeline running mode: TEST, PROD or GRPC.')


class BasePipeline(object):
    """Fueling base pipeline."""
    SPARK_CONTEXT = None

    def __init__(self, name):
        """Pipeline constructor."""
        # Values constructed on driver and broadcast to executors.
        self.name = name
        self.FLAGS = None
        # Values constructed on driver or on demand.
        self._mongo = None
        # Values constructed on driver and not shared.
        BasePipeline.SPARK_CONTEXT = SparkContext.getOrCreate(SparkConf().setAppName(self.name))

    def run_test(self):
        """Run the pipeline in test mode."""
        raise Exception('{}::run_test not implemented!'.format(self.name))

    def run_prod(self):
        """Run the pipeline in production mode."""
        raise Exception('{}::run_prod not implemented!'.format(self.name))

    def run_grpc(self):
        """Run the pipeline in GRPC mode."""
        raise Exception('{}::run_grpc not implemented!'.format(self.name))

    # Helper functions.
    @classmethod
    def context(cls):
        """Get the SparkContext."""
        if cls.SPARK_CONTEXT is None:
            cls.SPARK_CONTEXT = SparkContext.getOrCreate(SparkConf().setAppName('BasePipeline'))
        return cls.SPARK_CONTEXT

    def mongo(self):
        """Get a mongo instance."""
        if self._mongo is None:
            self._mongo = Mongo(self.FLAGS)
        return self._mongo

    def __main__(self, argv):
        """Run the pipeline."""
        self.FLAGS = flags.FLAGS.flag_values_dict()
        mode = self.FLAGS['running_mode']
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
        app.run(self.__main__)
