#!/usr/bin/env python
"""Fueling base pipeline."""

from datetime import datetime
import os
import sys
import traceback

from absl import app
from absl import flags
from absl.testing import absltest
from pyspark import SparkConf, SparkContext

from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
from fueling.common.storage.filesystem import Filesystem
import fueling.common.logging as logging


# Pipeline lifecycle:
# BasePipeline.main() -> BasePipeline.__main__() -> SparkSubmitterClient().submit()
#                                |                              |
#                                v (at local)                   |(in cloud)
#                        BasePipeline.init() <-------------------
#                                |
#                                v
#                        YourPipeline.run()  # The only thing you should override
#                                |
#                                v
#                        BasePipeline.stop()
class BasePipelineV2(object):
    """Fueling base pipeline."""
    def run(self):
        """Run the pipeline."""
        raise Exception('Not implemented!')

    # Helper functions.
    def to_rdd(self, data):
        """Get an RDD of data."""
        if BasePipelineV2.SPARK_CONTEXT is None:
            logging.fatal('Pipeline not inited. Please run init() first.')
        return BasePipelineV2.SPARK_CONTEXT.parallelize(data)

    def is_test(self):
        return self.FLAGS.get('running_mode') == 'TEST'

    def our_storage(self):
        """Get a BOS client if in PROD mode, or local filesystem if in TEST mode."""
        return Filesystem() if self.is_test() else BosClient()

    @staticmethod
    def is_partner_job():
        """Test if it's partner's job."""
        return os.environ.get('PARTNER_BOS_REGION')

    @staticmethod
    def partner_storage():
        """Get partner's storage instance."""
        if os.environ.get('PARTNER_BOS_REGION'):
            is_partner = True
            return BosClient(is_partner)
        return None

    # Internal members.
    SPARK_CONTEXT = None

    def init(self):
        """Init necessary infra."""
        if BasePipelineV2.SPARK_CONTEXT is None:
            spark_conf = SparkConf().setAppName(self.__class__.__name__)
            if flags.FLAGS.cpu > 1:
                spark_conf.setMaster(F'local[{flags.FLAGS.cpu}]')
            BasePipelineV2.SPARK_CONTEXT = SparkContext.getOrCreate(spark_conf)
        FLAGS = flags.FLAGS
        if not FLAGS.job_id:
            FLAGS.job_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # Member variables are available on both driver and executors.
        self.FLAGS = FLAGS.flag_values_dict()
        logging.info('Running job with owner={}, id={}'.format(FLAGS.job_owner, FLAGS.job_id))

    @classmethod
    def stop(cls):
        """Stop the pipeline."""
        if cls.SPARK_CONTEXT is not None:
            cls.SPARK_CONTEXT.stop()
            cls.SPARK_CONTEXT = None

    def __main__(self, argv):
        """Run the pipeline."""
        if flags.FLAGS.cloud:
            SparkSubmitterClient(self.entrypoint).submit()
            return
        try:
            self.init()
            self.run()
        finally:
            self.stop()

    def main(self):
        """Kick off everything."""
        self.entrypoint = traceback.extract_stack(limit=2)[0].filename
        app.run(self.__main__)


# SequentialPipeline lifecycle:
# BasePipeline.main() -> BasePipeline.__main__() -> SparkSubmitterClient().submit()
#                                |                              |
#                                V (at local)                   |(in cloud)
#                        SequentialPipeline.init() <-------------
#                                |
#                                V
#                        SequentialPipeline.run()
#                        --------|-----------
#                        |       |          |
#                        V       V          V
#          pipeline1.run() pipeline2.run() ...
#                        ---------------------
#                                |
#                                V
#                        BasePipeline.stop()
class SequentialPipelineV2(BasePipelineV2):
    """
    A sequential of sub-pipelines. Run it like
        SequentialPipeline([
            Pipeline1(),
            Pipeline2(arg),
            Pipeline3(),
        ]).main()
    """

    def __init__(self, phases):
        self.phases = phases

    def init(self):
        """Init all sub-pipelines."""
        super().init()
        for phase in self.phases:
            phase.init()

    def run(self):
        """Run the pipeline in test mode."""
        for phase in self.phases:
            phase.run()


# PipelineTest lifecycle:
# BasePipelineTest.main() -> BasePipelineTest.setUp() -> BasePipeline.init()
#                                                                 |
#                                    ------------------------------
#                                    |           |                |
#                                    V           V                V
#             YourPipelineTest.test_A() YourPipelineTest.test_B() ...
#                                    ------------------------------
#                                                |
#                                                V
#                             BasePipelineTest.tearDown() -> BasePipeline.stop()
# Be careful about the side effect of test functions.
class BasePipelineTest(absltest.TestCase):

    def setUp(self, pipeline):
        super().setUp()
        self.pipeline = pipeline
        self.pipeline.init()

    def tearDown(self):
        self.pipeline.stop()
        super().tearDown()
