#!/usr/bin/env python
"""BAE proxy."""

from http import HTTPStatus
import json
import os

from absl import app as absl_app
from absl import logging
import flask
import flask_restful
import google.protobuf.json_format as json_format

from spark_submit_arg_pb2 import SparkSubmitArg, Env


class SparkSubmitJob(flask_restful.Resource):
    """SparkSubmit job restful service"""

    def post(self):
        """Accept user request, verify and process."""
        try:
            request = flask.request.get_json()
            parser = json_format.Parse if isinstance(request, str) else json_format.ParseDict
            arg = parser(request, SparkSubmitArg())
            http_code, msg = SparkSubmitJob.spark_submit(arg)
        except json_format.ParseError:
            http_code = HTTPStatus.BAD_REQUEST
            msg = 'SparkSubmitArg format error!'
        return json.dumps({'message': msg}), http_code

    @staticmethod
    def spark_submit(arg):
        """Submit job."""
        # Configs.
        K8S_MASTER = 'https://180.76.150.16:6443'
        STORAGE = '/mnt/bos'
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
            'LOG_VERBOSITY': Env.LogVerbosity.Name(arg.log_verbosity),
        }

        # Prepare fueling package.
        fueling_zip_path = arg.job.fueling_zip_path
        if arg.job.fueling_zip_base64:
            # Decode, upload.
            # fueling_zip_path = xxx
            pass
        ENVS['APOLLO_FUELING_PYPATH'] = fueling_zip_path

        # Partner storage.
        if arg.partner.storage_writable:
            ENVS['PARTNER_STORAGE_WRITABLE'] = 'TRUE'
        if arg.partner.bos.bucket:
            ENVS.update({
                'PARTNER_BOS_REGION': arg.partner.bos.region,
                'PARTNER_BOS_BUCKET': arg.partner.bos.bucket,
                'PARTNER_BOS_ACCESS': arg.partner.bos.access,
                'PARTNER_BOS_SECRET': arg.partner.bos.secret,
            })
        if arg.partner.blob.storage_account:
            ENVS.update({
                'AZURE_STORAGE_ACCOUNT': arg.partner.blob.storage_account,
                'AZURE_STORAGE_ACCESS_KEY': arg.partner.blob.storage_access_key,
                'AZURE_BLOB_CONTAINER': arg.partner.blob.blob_container,
            })

        # Construct arguments.
        confs = (
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
                'docker_image': arg.docker_image,
                'workers': arg.worker.count,
                'worker_memory': arg.worker.memory,
                'worker_cpu': arg.worker.cpu,
                'worker_disk': arg.worker.disk,
            })

        # Select nodes.
        if arg.node_selector in {Env.NodeSelector.CPU, Env.NodeSelector.GPU}:
            confs += '--conf spark.kubernetes.node.selector.computetype={} '.format(
                Env.NodeSelector.Name(arg.node_selector))

        # Envs.
        for k, v in ENVS:
            confs += ('--conf spark.kubernetes.driverEnv.%(key)s=%(value)s '
                      '--conf spark.executorEnv.%(key)s=%(value)s ' % {
                          key: k, value: v})
        for k, v in SECRET_ENVS:
            confs += ('--conf spark.kubernetes.driver.secretKeyRef.%(key)s=%(value)s '
                      '--conf spark.kubernetes.executor.secretKeyRef.%(key)s=%(value)s ' % {
                          key: k, value: v})

        # TODO: Construct job name.
        job_name = 'TODO'  # job-id_job-owner_job-filename_timestamp
        cmd = ('spark-submit '
               '--deploy-mode cluster '
               '--master %(k8s_master)s '
               '--name %(job_name)s '
               '%(confs)s '
               '"%(entrypoint)s" '
               '%(flags)s ' % {
                   'k8s_master': K8S_MASTER,
                   'job_name': job_name,
                   'confs': confs,
                   'entrypoint': entrypoint,
                   'flags': '--running_mode=PROD ' + ' '.join(arg.flags),
               })
        logging.info('SHELL > {}'.format(cmd))

        # Execute command.
        #os.system(cmd)
        return HTTPStatus.OK, 'Job submitted!'


app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(SparkSubmitJob, '/')


def main(argv):
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == '__main__':
    absl_app.run(main)
