#!/usr/bin/env python
"""Spark submitter."""

from datetime import datetime
from http import HTTPStatus
import base64
import json
import os
import site
import subprocess
import threading

from absl import app
from absl import flags
from absl import logging
import boto3
import botocore
import flask
import flask_restful
import google.protobuf.json_format as json_format

from apps.k8s.spark_submitter.spark_submit_arg_pb2 import Env, JobRecord, SparkSubmitArg
from fueling.common.mongo_utils import Mongo
import fueling.common.proto_utils as proto_utils


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')


class SparkSubmitJob(flask_restful.Resource):
    """SparkSubmit job restful service"""

    def get(self):
        """Get job status."""
        job_id = flask.request.args.get('job_id')
        cmd = "kubectl get pods | grep %s | grep driver | awk '{print $3}'" % job_id
        status = (subprocess.check_output(cmd, shell=True) or b'Preparing').decode('ASCII').strip()
        return json.dumps({'status': status}), HTTPStatus.OK

    def post(self):
        """Accept user request, verify and process."""
        try:
            arg = json_format.Parse(flask.request.get_json(), SparkSubmitArg())
            job_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
            if flags.FLAGS.debug:
                self.spark_submit(job_id, arg)
            else:
                threading.Thread(target=self.spark_submit, args=(job_id, arg)).start()
            return json.dumps({'job_id': job_id}), HTTPStatus.OK
        except json_format.ParseError:
            return json.dumps({'error': 'Bad SparkSubmitArg format!'}), HTTPStatus.BAD_REQUEST

    @staticmethod
    def spark_submit(job_id, arg):
        """Submit job."""
        # Configs.
        K8S_MASTER = 'k8s://https://180.76.150.16:6443'
        STORAGE = '/mnt/bos'
        EXTRACTED_PATH = '/apollo/modules/data/fuel'
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
            'APOLLO_CONDA_ENV': arg.env.conda_env,
            'APOLLO_EXECUTORS': arg.worker.count,
            'LOG_VERBOSITY': Env.LogVerbosity.Name(arg.env.log_verbosity),
        }

        submitter = arg.user.submitter
        # running_role = arg.user.running_role
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
        if not flags.FLAGS.debug:
            job_record = JobRecord(id=job_id, arg=arg)
            Mongo().job_collection().insert_one(proto_utils.pb_to_dict(job_record))

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
        if arg.partner.blob.storage_account:
            ENVS.update({
                'AZURE_STORAGE_ACCOUNT': arg.partner.blob.storage_account,
                'AZURE_STORAGE_ACCESS_KEY': arg.partner.blob.storage_access_key,
                'AZURE_BLOB_CONTAINER': arg.partner.blob.blob_container,
            })

        # Construct arguments.
        confs = (
            # Overall
            '--conf spark.kubernetes.memoryOverheadFactor=0 '
            # Docker
            '--conf spark.kubernetes.container.image.pullPolicy=Always '
            '--conf spark.kubernetes.container.image.pullSecrets=baidubce '
            '--conf spark.kubernetes.container.image=%(docker_image)s '
            # Driver
            '--conf spark.driver.memory=2g '
            '--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark '
            # Executor
            '--conf spark.executor.instances=%(workers)s '
            '--conf spark.default.parallelism=%(workers)s '
            '--conf spark.executor.memory=%(worker_memory)sg '
            '--conf spark.kubernetes.executor.request.cores=%(worker_cpu)s '
            '--conf spark.kubernetes.executor.ephemeralStorageGB=%(worker_disk)s ' % {
                'docker_image': arg.env.docker_image,
                'workers': arg.worker.count,
                'worker_memory': arg.worker.memory,
                'worker_cpu': arg.worker.cpu,
                'worker_disk': arg.worker.disk,
            })

        # Select nodes.
        if arg.env.node_selector in {Env.NodeSelector.CPU, Env.NodeSelector.GPU}:
            confs += '--conf spark.kubernetes.node.selector.computetype={} '.format(
                Env.NodeSelector.Name(arg.env.node_selector))

        # Envs.
        for key, value in ENVS.items():
            confs += (f'--conf spark.kubernetes.driverEnv.{key}={value} '
                      f'--conf spark.executorEnv.{key}={value} ')
        for key, value in SECRET_ENVS.items():
            confs += (f'--conf spark.kubernetes.driver.secretKeyRef.{key}={value} '
                      f'--conf spark.kubernetes.executor.secretKeyRef.{key}={value} ')

        job_filename = os.path.splitext(os.path.basename(arg.job.entrypoint))[0].replace('_', '-')
        job_name = f'job-{job_id}-{submitter}-{job_filename}'
        site_package = site.getsitepackages()[0]
        if arg.job.entrypoint.endswith('.zip'):
            entrypoint = fueling_zip_path
        else:
            entrypoint = os.path.join(EXTRACTED_PATH, arg.job.entrypoint)
        cmd = (f'{site_package}/pyspark/bin/spark-submit --deploy-mode cluster '
               f'--master {K8S_MASTER} --name {job_name} {confs} '
               f'"{entrypoint}" --running_mode=PROD {arg.job.flags} '
               f'--job_owner="{submitter}" --job_id="{job_id}"')
        # Execute command.
        logging.debug('SHELL > {}'.format(cmd))
        os.system(cmd)


flask_app = flask.Flask(__name__)
api = flask_restful.Api(flask_app)
api.add_resource(SparkSubmitJob, '/')


def main(argv):
    if flags.FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    flask_app.run(host='0.0.0.0', port=8000, debug=flags.FLAGS.debug)


if __name__ == '__main__':
    app.run(main)
