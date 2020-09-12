#!/usr/bin/env python
"""Spark submitter utils."""

import base64
import os
import site

import boto3
import botocore

from apps.k8s.spark_submitter.spark_submit_arg_pb2 import Env, JobRecord
from fueling.common.mongo_utils import Mongo
import fueling.common.job_utils as job_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


class Utils(object):

    @staticmethod
    def spark_submit(job_id, arg, record_job=True):
        """Submit job."""
        # Configs.
        K8S_MASTER = 'k8s://https://180.76.60.99:6443'
        STORAGE = '/mnt/bos'
        EXTRACTED_PATH = '/apollo/modules/data/fuel'
        SPARK_LOG_STORAGE = 'file:///mnt/bos/modules/data/spark'
        EVENTLOG_DIR = os.path.join(SPARK_LOG_STORAGE, 'spark-events')
        SECRET_ENVS = {
            'BOS_ACCESS': 'bos-secret:ak',
            'BOS_SECRET': 'bos-secret:sk',
            'BOS_BUCKET': 'bos-secret:bucket',
            'BOS_REGION': 'bos-secret:region',
            'MONGO_USER': 'mongo-secret:user',
            'MONGO_PASSWD': 'mongo-secret:passwd',
            'OUTLOOK_USER': 'outlook-secret:user',
            'OUTLOOK_PASSWD': 'outlook-secret:passwd',
            'REDIS_PASSWD': 'redis-secret:passwd',
        }
        ENVS = {
            'APOLLO_EXECUTORS': arg.worker.count,
            'LOG_VERBOSITY': Env.LogVerbosity.Name(arg.env.log_verbosity),
        }

        running_role = arg.user.running_role
        submitter = running_role or arg.user.submitter
        # TODO: Verify submitter and running_role, and save to DB for job management.

        # Prepare fueling package.
        fueling_zip_path = arg.job.fueling_zip_path
        if arg.job.fueling_zip_base64:
            if arg.job.entrypoint.endswith('.zip'):
                # Bazel-built zip app.
                zip_app = os.path.basename(arg.job.entrypoint)[:-4] + '.py'
                fueling_zip_key = os.path.join('modules/data/jobs', job_id, zip_app)
            else:
                fueling_zip_key = os.path.join('modules/data/jobs', job_id, 'fueling.zip')
            boto3.client(
                's3',
                endpoint_url='http://s3.{}.bcebos.com'.format(os.environ.get('BOS_REGION')),
                region_name=os.environ.get('BOS_REGION'),
                config=botocore.client.Config(signature_version='s3v4')
            ).put_object(
                Bucket=os.environ.get('BOS_BUCKET'),
                Body=base64.b64decode(arg.job.fueling_zip_base64),
                Key=fueling_zip_key)
            fueling_zip_path = os.path.join(STORAGE, fueling_zip_key)
            # Only store hash to save space.
            arg.job.fueling_zip_base64 = str(hash(arg.job.fueling_zip_base64))
        ENVS['APOLLO_FUELING_PYPATH'] = fueling_zip_path

        # Update job database.
        if record_job:
            job_record = JobRecord(id=job_id, arg=arg)
            Mongo().job_collection().insert_one(proto_utils.pb_to_dict(job_record))
            jobUtils = job_utils.JobUtils(job_id)
            jobUtils.save_job_submit_info()
            if arg.partner.job_type:
                jobUtils.save_job_type(arg.partner.job_type)
            if arg.partner.vehicle_sn:
                jobUtils.save_job_vehicle_sn(arg.partner.vehicle_sn)

        # Partner storage.
        if arg.partner.storage_writable:
            ENVS['PARTNER_STORAGE_WRITABLE'] = 'TRUE'
        if arg.partner.bos.bucket:
            ENVS.update({
                'PARTNER_BOS_REGION': arg.partner.bos.region,
                'PARTNER_BOS_BUCKET': arg.partner.bos.bucket,
                'PARTNER_BOS_ACCESS': arg.partner.bos.access_key,
                'PARTNER_BOS_SECRET': arg.partner.bos.secret_key,
            })
        if arg.partner.email:
            ENVS['PARTNER_EMAIL'] = arg.partner.email
        if arg.partner.vehicle_sn:
            ENVS['PARTNER_VEHICLE_SN'] = arg.partner.vehicle_sn

        # Construct arguments.
        confs = (
            # Overall
            '--conf spark.kubernetes.memoryOverheadFactor=0 '
            '--conf spark.kubernetes.executor.volumes.emptyDir.shm.mount.path=/dev/shm '
            '--conf spark.kubernetes.executor.volumes.emptyDir.shm.options.medium=Memory '
            '--conf yarn.log-aggregation-enable=true '
            '--conf spark.eventLog.enabled=true '
            f'--conf spark.eventLog.dir={EVENTLOG_DIR} '
            # Docker
            '--conf spark.kubernetes.container.image.pullPolicy=Always '
            '--conf spark.kubernetes.container.image.pullSecrets=baidubce '
            f'--conf spark.kubernetes.container.image={arg.env.docker_image} '
            # Driver
            f'--conf spark.driver.memory={arg.driver.driver_memory}g '
            '--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark '
            # Executor
            f'--conf spark.kubernetes.executor.deleteOnTermination=false '
            f'--conf spark.executor.instances={arg.worker.count} '
            f'--conf spark.default.parallelism={arg.worker.count} '
            f'--conf spark.executor.memory={arg.worker.memory}g '
            f'--conf spark.kubernetes.executor.request.cores={arg.worker.cpu} '
            f'--conf spark.kubernetes.executor.gpus={arg.worker.gpu} '
            f'--conf spark.kubernetes.executor.ephemeralStorageGB={arg.worker.disk} ')

        # Use node-selector for different types of jobs
        if arg.env.node_selector != Env.NodeSelector.ANY:
            confs += '--conf spark.kubernetes.node.selector.{}=YES '.format(
                Env.NodeSelector.Name(arg.env.node_selector))

        # Envs.
        for key, value in ENVS.items():
            confs += (f'--conf spark.kubernetes.driverEnv.{key}={value} '
                      f'--conf spark.executorEnv.{key}={value} ')
        for key, value in SECRET_ENVS.items():
            confs += (f'--conf spark.kubernetes.driver.secretKeyRef.{key}={value} '
                      f'--conf spark.kubernetes.executor.secretKeyRef.{key}={value} ')

        # Rapids.
        if arg.rapids.rapids_enabled:
            discovery_script = arg.rapids.rapids_discovery_script
            concurrent_tasks = arg.rapids.rapids_concurrent_tasks
            executor_gpus = arg.rapids.rapids_executor_gpu
            confs += (f'--conf spark.rapids.memory.pinnedPool.size=2G '
                      f'--conf spark.sql.files.maxPartitionBytes=512m '
                      f'--conf spark.plugins=com.nvidia.spark.SQLPlugin '
                      f'--conf spark.executor.resource.gpu.vendor=nvidia.com '
                      f'--conf spark.task.cpus=1 '
                      f'--conf spark.executor.resource.gpu.discoveryScript={discovery_script} '
                      f'--conf spark.rapids.sql.concurrentGpuTasks={concurrent_tasks} '
                      f'--conf spark.task.resource.gpu.amount={arg.rapids.rapids_task_gpu} '
                      f'--conf spark.executor.cores={arg.rapids.rapids_task_cores} '
                      f'--conf spark.executor.resource.gpu.amount={executor_gpus} ')

        job_filename = os.path.splitext(os.path.basename(arg.job.entrypoint))[0].replace('_', '-')
        job_name = f'job-{job_id}-{submitter}-{job_filename}'
        site_package = site.getsitepackages()[0]
        if arg.job.entrypoint.endswith('.zip'):
            entrypoint = fueling_zip_path
        else:
            entrypoint = os.path.join(EXTRACTED_PATH, arg.job.entrypoint)
        cmd = (f'{site_package}/pyspark/bin/spark-submit --deploy-mode cluster '
               f'--master {K8S_MASTER} --name {job_name} {confs} '
               f'"{entrypoint}" {arg.job.flags} --job_owner="{submitter}" --job_id="{job_id}"')
        # Execute command.
        logging.debug('SHELL > {}'.format(cmd))
        os.system(cmd)
