#!/usr/bin/env python
"""Fueling base pipeline."""

from datetime import datetime
import os
import sys

from absl import app, flags
from pyspark import SparkConf, SparkContext
import colored_glog as glog

from fueling.common.mongo_utils import Mongo
import fueling.common.storage.blob_client as blob_client
import fueling.common.storage.bos_client as bos_client


flags.DEFINE_string('running_mode', None, 'Pipeline running mode: TEST, PROD or GRPC.')
flags.DEFINE_boolean('debug', False, 'Enable debug logging.')

flags.DEFINE_string('job_owner', 'apollo', 'Pipeline job owner.')
flags.DEFINE_string('job_id', None, 'Pipeline job ID.')


class BasePipeline(object):
    """Fueling base pipeline."""
    SPARK_CONTEXT = None

    def __init__(self, name):
        """Pipeline constructor."""
        # Values constructed on driver and broadcast to executors.
        self.name = name
        self.FLAGS = None
        # Values constructed on driver and not shared.
        BasePipeline.SPARK_CONTEXT = SparkContext.getOrCreate(SparkConf().setAppName(self.name))

    def run_test(self):
        """Run the pipeline in test mode."""
        raise Exception('{}::run_test not implemented!'.format(self.name))

    def run_prod(self):
        """Run the pipeline in production mode."""
        raise Exception('{}::run_prod not implemented!'.format(self.name))

    # Helper functions.
    @classmethod
    def context(cls):
        """Get the SparkContext."""
        if cls.SPARK_CONTEXT is None:
            cls.SPARK_CONTEXT = SparkContext.getOrCreate(SparkConf().setAppName('BasePipeline'))
        return cls.SPARK_CONTEXT

    def to_rdd(self, data):
        """Get an RDD of data."""
        return self.context().parallelize(data)

    def mongo(self):
        """Get a mongo client."""
        return Mongo(self.FLAGS)

    # TODO(xiaoxq): Retire later.
    def bos(self):
        """Get a BOS client."""
        return bos_client.BosClient()

    @staticmethod
    def has_partner():
        """Test if we have partner bos."""
        return os.environ.get('PARTNER_BOS_REGION') or os.environ.get('AZURE_STORAGE_ACCOUNT')

    @staticmethod
    def partner_object_storage():
        """Get partner's object storage client."""
        if os.environ.get('PARTNER_BOS_REGION'):
            is_partner = True
            return bos_client.BosClient(is_partner)
        elif os.environ.get('AZURE_STORAGE_ACCOUNT'):
            return blob_client.BlobClient()
        return None

    def __main__(self, argv):
        """Run the pipeline."""
        self.FLAGS = flags.FLAGS.flag_values_dict()
        if self.FLAGS.get('debug'):
            glog.setLevel(glog.DEBUG)

        mode = self.FLAGS.get('running_mode')
        if mode is None:
            glog.fatal('No running mode is specified! Please run the pipeline with /tools/<runner> '
                       'instead of calling native "python".')
            sys.exit(1)
        if not self.FLAGS.get('job_id'):
            self.FLAGS['job_id'] = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        glog.info('Running {} job in {} mode, owner={}, id={}'.format(
            self.name, mode, self.FLAGS.get('job_owner'), self.FLAGS.get('job_id')))
        if mode == 'TEST':
            self.run_test()
        else:
            self.run_prod()

    def main(self):
        app.run(self.__main__)
