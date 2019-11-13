#!/usr/bin/env python
"""Fueling base pipeline."""

from datetime import datetime
import os
import sys

from absl import app
from absl import flags
from pyspark import SparkConf, SparkContext

import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client


flags.DEFINE_string('running_mode', None, 'Pipeline running mode: TEST, PROD or GRPC.')
flags.DEFINE_string('job_owner', 'apollo', 'Pipeline job owner.')
flags.DEFINE_string('job_id', None, 'Pipeline job ID.')


class BasePipeline(object):
    """Fueling base pipeline."""
    # Class variables are only available on drivers.
    SPARK_CONTEXT = None

    def init(self):
        """Should be called explicitly after app inited."""
        # Member variables are available on both driver and executors.
        self.FLAGS = flags.FLAGS.flag_values_dict()

    def run_test(self):
        """Run the pipeline in test mode."""
        raise Exception('Not implemented!')

    def run_prod(self):
        """Run the pipeline in production mode."""
        raise Exception('Not implemented!')

    # Helper functions.
    def to_rdd(self, data):
        """Get an RDD of data."""
        return BasePipeline.SPARK_CONTEXT.parallelize(data)

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
            import fueling.common.storage.blob_client as blob_client
            return blob_client.BlobClient()
        return None

    def __main__(self, argv):
        """Run the pipeline."""
        BasePipeline.SPARK_CONTEXT = SparkContext.getOrCreate(
            SparkConf().setAppName(self.__class__.__name__))
        mode = flags.FLAGS.running_mode
        if mode is None:
            logging.fatal('No running mode is specified! Please run the pipeline with either\n'
                          '    tools/submit-job-to-local.sh\n'
                          '    tools/submit-job-to-k8s.py')
            sys.exit(1)
        if not flags.FLAGS.job_id:
            flags.FLAGS.job_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        logging.info('Running job in {} mode, owner={}, id={}'.format(
            mode, flags.FLAGS.job_owner, flags.FLAGS.job_id))

        self.init()
        if mode == 'TEST':
            self.run_test()
        else:
            self.run_prod()
        BasePipeline.SPARK_CONTEXT.stop()

    def main(self):
        """Kick off everything."""
        app.run(self.__main__)


class SequentialPipeline(BasePipeline):
    """
    A sequential of sub-pipelines. Run it like
        SequentialPipeline([
            Pipeline1(),
            Pipeline2(arg1),
            Pipeline3(),
        ]).main()
    """

    def __init__(self, phases):
        self.phases = phases

    def init(self):
        """Init all sub-pipelines."""
        BasePipeline.init(self)
        for phase in self.phases:
            phase.init()

    def run_test(self):
        """Run the pipeline in test mode."""
        for phase in self.phases:
            phase.run_test()

    def run_prod(self):
        """Run the pipeline in production mode."""
        for phase in self.phases:
            phase.run_prod()
