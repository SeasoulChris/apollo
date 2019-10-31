#!/usr/bin/env python
"""Fueling base pipeline."""

from datetime import datetime
import os
import sys

from absl import app
from absl import flags
from pyspark import SparkConf, SparkContext

import fueling.common.logging as logging
import fueling.common.storage.blob_client as blob_client
import fueling.common.storage.bos_client as bos_client


flags.DEFINE_string('running_mode', None, 'Pipeline running mode: TEST, PROD or GRPC.')
flags.DEFINE_string('job_owner', 'apollo', 'Pipeline job owner.')
flags.DEFINE_string('job_id', None, 'Pipeline job ID.')


class BasePipeline(object):
    """Fueling base pipeline."""
    SPARK_CONTEXT = None

    def __init__(self):
        """Pipeline constructor."""
        # Values constructed on driver and broadcast to executors.
        self.name = self.__class__.__name__
        self.FLAGS = None
        # Values constructed on driver and not shared.
        BasePipeline.SPARK_CONTEXT = SparkContext.getOrCreate(SparkConf().setAppName(self.name))

    def run_test(self):
        """Run the pipeline in test mode."""
        raise Exception('Not implemented!')

    def run_prod(self):
        """Run the pipeline in production mode."""
        raise Exception('Not implemented!')

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
        mode = self.FLAGS.get('running_mode')
        if mode is None:
            logging.fatal('No running mode is specified! Please run the pipeline with '
                          './tools/submit-job-to-xxx instead of calling native "python".')
            sys.exit(1)
        if not self.FLAGS.get('job_id'):
            self.FLAGS['job_id'] = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        logging.info('Running {} job in {} mode, owner={}, id={}'.format(
            self.name, mode, self.FLAGS.get('job_owner'), self.FLAGS.get('job_id')))
        if mode == 'TEST':
            self.run_test()
        else:
            self.run_prod()
        self.context().stop()

    def main(self):
        app.run(self.__main__)


class SequentialPipeline(BasePipeline):
    def __init__(self, phases):
        """Pipeline constructor."""
        BasePipeline.__init__(self)
        self.phases = [phase() for phase in phases]

    def run_test(self):
        """Run the pipeline in test mode."""
        for phase in self.phases:
            phase.run_test()

    def run_prod(self):
        """Run the pipeline in production mode."""
        for phase in self.phases:
            phase.run_prod()
