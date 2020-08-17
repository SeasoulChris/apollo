#!/usr/bin/env python
"""Fueling base pipeline."""

from datetime import datetime
from datetime import timezone
import dateutil.parser
import json
import os
import sys
import time
import traceback

from absl import app
from absl import flags
from absl.testing import absltest
from pyspark import SparkConf, SparkContext

from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.common.job_utils import JobUtils
from fueling.common.kubectl_utils import Kubectl
from fueling.common.mongo_utils import Mongo
from fueling.common.storage.bos_client import BosClient
from fueling.common.storage.filesystem import Filesystem
import fueling.common.context_utils as context_utils
import fueling.common.logging as logging


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
        context = BasePipeline.SPARK_CONTEXT
        if context is None:
            logging.fatal('Pipeline not inited. Please run init() first.')
        # If we got more than 100k elements, slice them to (100*N) partitions.
        count = len(data)
        slices = None
        if count > 100000:
            slices = count // context.defaultParallelism // 100
            logging.info(F'Slice {count} elements to {slices} slices.')
        return context.parallelize(data, slices)

    def our_storage(self):
        """Get a BOS client if in PROD mode, local filesystem if in LOCAL mode,
        or local Bazel test filesystem if in TEST mode."""
        return Filesystem() if context_utils.is_local() else BosClient()

    @staticmethod
    def is_partner_job():
        """Test if it's partner's job."""
        return os.environ.get('PARTNER_VEHICLE_SN')

    @staticmethod
    def partner_storage():
        """Get partner's storage instance."""
        if os.environ.get('PARTNER_VEHICLE_SN'):
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

    def __cloud_job_post_process__(self, job_failed=False):
        kubectl = Kubectl()
        pod_name_pattern = F'job-{flags.FLAGS.job_id}-*'
        pod_list = kubectl.get_pods_by_pattern(pod_name_pattern)
        if len(pod_list) > 0:
            for pod in pod_list:
                pod_name = pod.metadata.name
                pod_namespace = pod.metadata.namespace
                if flags.FLAGS.auto_delete_driver_pod:
                    kubectl.delete_pod(pod_name, pod_namespace)
                else:
                    pod_desc = kubectl.describe_pod(pod_name, pod_namespace, tojson=True)
                    if job_failed:
                        pod_desc['status']['phase'] = phase = 'Failed'
                    else:
                        if pod_desc['status']['phase'] == 'Running':
                            pod_desc['status']['phase'] = phase = 'Succeeded'
                        else:
                            phase = pod_desc['status']['phase']
                    creation_timestamp = (dateutil.parser
                                          .parse(pod_desc['metadata']['creationTimestamp'])
                                          .replace(tzinfo=timezone.utc))
                    pod_log = None
                    # add retry in case the kubectl doesn't get log correctly
                    for i in range(5):
                        try:
                            pod_log = kubectl.logs(pod_name, pod_namespace)
                            logging.info('Querying log successfully')
                            break
                        except Exception as ex:
                            logging.info('Querying job log failed')
                            logging.error(f'{ex}')
                            logging.error(traceback.format_exc())
                            time.sleep(10)
                            continue
                    if pod_log is None:
                        continue
                    pod_desc_str = json.dumps(pod_desc, sort_keys=True, indent=4,
                                              separators=(', ', ': '))
                    pod_data = {'pod_name': pod_name, 'namespace': pod_namespace, 'phase': phase,
                                'desc': pod_desc_str, 'job_id': flags.FLAGS.job_id, 'logs': pod_log,
                                'creation_timestamp': creation_timestamp.timestamp()}
                    if pod_name.endswith('-driver'):
                        JobUtils(flags.FLAGS.job_id).save_job_phase(phase)
                        pod_data.update({'pod_type': 'driver'})
                    else:
                        pod_data.update({'pod_type': 'executor'})
                    Mongo().job_log_collection().insert_one(pod_data)
                    logging.info(F'Save {pod_name} log success')
        else:
            logging.info(F'Failed to find exact pod for "{pod_name_pattern}"')

    def __main__(self, argv):
        """Run the pipeline."""
        if flags.FLAGS.cloud:
            SparkSubmitterClient(self.entrypoint).submit()
            return
        try:
            if context_utils.is_cloud():
                JobUtils(flags.FLAGS.job_id).save_job_partner(bool(self.is_partner_job()))
        except Exception:
            logging.error('save job partner failed')
        job_failed = False
        try:
            self.init()
            self.run()
        except Exception:
            job_failed = True
            logging.error(traceback.format_exc())
        finally:
            self.stop()
        if context_utils.is_cloud():
            self.__cloud_job_post_process__(job_failed)
        if job_failed:
            sys.exit(1)

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
