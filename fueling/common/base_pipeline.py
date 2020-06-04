#!/usr/bin/env python
"""Fueling base pipeline."""

from datetime import datetime
from datetime import timezone
import dateutil.parser
import json
import os
import sys
import traceback

from absl import app
from absl import flags
from absl.testing import absltest
from pyspark import SparkConf, SparkContext

from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.common.kubectl_utils import Kubectl
from fueling.common.mongo_utils import Mongo
from fueling.common.storage.bazel_filesystem import BazelFilesystem
from fueling.common.storage.bos_client import BosClient
from fueling.common.storage.filesystem import Filesystem
import fueling.common.logging as logging


flags.DEFINE_string('running_mode', 'LOCAL', 'Pipeline running mode: TEST, LOCAL, PROD.')
flags.DEFINE_string('job_owner', 'apollo', 'Pipeline job owner.')
flags.DEFINE_string('job_id', None, 'Pipeline job ID.')
flags.DEFINE_string('input_data_path', None, 'Input data path which is commonly used by pipelines.')
flags.DEFINE_string(
    'output_data_path',
    None,
    'Output data path which is commonly used by pipelines.')
flags.DEFINE_boolean('auto_delete_driver_pod', False, 'Auto-delete driver pod when finish.')


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
class BasePipeline(object):
    """Fueling base pipeline."""

    def run(self):
        """Run the pipeline."""
        raise Exception('Not implemented!')

    # Helper functions.
    def to_rdd(self, data):
        """Get an RDD of data."""
        if BasePipeline.SPARK_CONTEXT is None:
            logging.fatal('Pipeline not inited. Please run init() first.')
        return BasePipeline.SPARK_CONTEXT.parallelize(data)

    def is_test(self):
        return self.FLAGS.get('running_mode') == 'TEST'

    def is_local(self):
        return self.FLAGS.get('running_mode') == 'LOCAL'

    def our_storage(self):
        """Get a BOS client if in PROD mode, local filesystem if in LOCAL mode,
        or local Bazel test filesystem if in TEST mode."""
        if self.is_test():
            return BazelFilesystem()
        return Filesystem() if self.is_local() else BosClient()

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
        if BasePipeline.SPARK_CONTEXT is None:
            spark_conf = SparkConf().setAppName(self.__class__.__name__)
            if flags.FLAGS.cpu > 1:
                spark_conf.setMaster(F'local[{flags.FLAGS.cpu}]')
            BasePipeline.SPARK_CONTEXT = SparkContext.getOrCreate(spark_conf)
            # Mute general Spark INFO logs.
            BasePipeline.SPARK_CONTEXT.setLogLevel('WARN')
        FLAGS = flags.FLAGS
        if not FLAGS.job_id:
            FLAGS.job_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # Member variables are available on both driver and executors.
        self.FLAGS = FLAGS.flag_values_dict()
        logging.info('Running job with owner={}, id={}'.format(FLAGS.job_owner, FLAGS.job_id))

    @staticmethod
    def stop():
        """Stop the pipeline."""
        if BasePipeline.SPARK_CONTEXT is not None:
            BasePipeline.SPARK_CONTEXT.stop()
            BasePipeline.SPARK_CONTEXT = None

    def __cloud_job_post_process__(self):
        kubectl = Kubectl()
        driver_pod_name_pattern = F'job-{flags.FLAGS.job_id}-*-driver'
        driver_pod = kubectl.get_pods_by_pattern(driver_pod_name_pattern)
        if len(driver_pod) == 1:
            pod_name = driver_pod[0].metadata.name
            pod_namespace = driver_pod[0].metadata.namespace
            if flags.FLAGS.auto_delete_driver_pod:
                kubectl.delete_pod(pod_name, pod_namespace)
            else:
                pod_log = kubectl.logs(pod_name, pod_namespace)
                pod_desc = kubectl.describe_pod(pod_name, pod_namespace, tojson=True)
                if pod_desc['status']['phase'] == 'Running':
                    phase = 'Succeeded'
                else:
                    phase = pod_desc['status']['phase']
                creation_timestamp = (dateutil.parser
                                      .parse(pod_desc['metadata']['creationTimestamp'])
                                      .replace(tzinfo=timezone.utc))
                Mongo().job_log_collection().insert_one(
                    {'logs': pod_log,
                     'desc': json.dumps(pod_desc, sort_keys=True, indent=4,
                                        separators=(', ', ': ')),
                     'phase': phase,
                     'job_id': flags.FLAGS.job_id,
                     'pod_name': pod_name,
                     'namespace': pod_namespace,
                     'creation_timestamp': creation_timestamp.timestamp()})
                logging.info(F'Save driver log success')
        else:
            logging.info(F'Failed to find exact driver pod for "{driver_pod_name_pattern}"')

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
        if flags.FLAGS.running_mode == 'PROD':
            self.__cloud_job_post_process__()

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
class SequentialPipeline(BasePipeline):
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

    """
    Note that setUp() and tearDown() are run once for each test method,
    instead of once for the entire test suite
    """

    def setUp(self, pipeline):
        super().setUp()
        self.pipeline = pipeline
        self.pipeline.init()

    def tearDown(self):
        self.pipeline.stop()
        super().tearDown()
