#!/usr/bin/env python
"""Fueling base pipeline."""

from datetime import datetime
import os
import sys
import traceback

from absl import app
from absl import flags
from pyspark import SparkConf, SparkContext

from fueling.common.internal.cloud_submitter import CloudSubmitter
from fueling.common.storage.bos_client import BosClient
from fueling.common.storage.filesystem import Filesystem
import fueling.common.logging as logging


flags.DEFINE_string('job_owner', 'apollo', 'Pipeline job owner.')
flags.DEFINE_string('job_id', None, 'Pipeline job ID.')
flags.DEFINE_string('input_data_path', None, 'Input data path which is commonly used by pipelines.')
flags.DEFINE_string('output_data_path', None, 'Output data path which is commonly used by pipelines.')

# TODO(xiaoxq): Retire in V2.
flags.DEFINE_string('running_mode', 'TEST', 'Pipeline running mode: TEST, PROD or GRPC.')


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

    def our_storage(self):
        """Get a BOS client if in PROD mode, or local filesystem if in TEST mode."""
        return Filesystem() if self.FLAGS.get('running_mode') == 'TEST' else BosClient()

    @staticmethod
    def is_partner_job():
        """Test if it's partner's job."""
        return os.environ.get('PARTNER_BOS_REGION') or os.environ.get('AZURE_STORAGE_ACCOUNT')

    @staticmethod
    def partner_storage():
        """Get partner's storage instance."""
        if os.environ.get('PARTNER_BOS_REGION'):
            is_partner = True
            return BosClient(is_partner)
        elif os.environ.get('AZURE_STORAGE_ACCOUNT'):
            import fueling.common.storage.blob_client as blob_client
            return blob_client.BlobClient()
        return None

    def __main__(self, argv):
        """Run the pipeline."""
        BasePipeline.SPARK_CONTEXT = SparkContext.getOrCreate(
            SparkConf().setAppName(self.__class__.__name__))
        mode = flags.FLAGS.running_mode
        if not flags.FLAGS.job_id:
            flags.FLAGS.job_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        logging.info('Running job in {} mode, owner={}, id={}'.format(
            mode, flags.FLAGS.job_owner, flags.FLAGS.job_id))

        try:
            self.init()
            if flags.FLAGS.cloud:
                CloudSubmitter(self.entrypoint).submit()
            elif mode == 'TEST':
                self.run_test()
            else:
                self.run_prod()
        finally:
            BasePipeline.SPARK_CONTEXT.stop()

    def main(self):
        """Kick off everything."""
        self.entrypoint = traceback.extract_stack(limit=2)[0].filename
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
