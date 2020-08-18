#!/usr/bin/env python
"""Spark submitter."""

from datetime import datetime
from http import HTTPStatus
import json
import subprocess
import threading
import traceback

from absl import app
from absl import flags
from absl import logging
import flask
import flask_restful
import google.protobuf.json_format as json_format

from apps.k8s.spark_submitter.job_processor import JobProcessor
from apps.k8s.spark_submitter.saas_job_arg_pb2 import SaasJobArg
from apps.k8s.spark_submitter.spark_submit_arg_pb2 import SparkSubmitArg
from apps.k8s.spark_submitter.utils import Utils


flags.DEFINE_boolean('debug', False, 'Enable debug mode.')

# Define the total resources here and reject jobs applying too many
# TODO(all): see if we can get these numbers dynamically by calling K8S interfaces
flags.DEFINE_integer('total_memory', 250 * 5, 'cluster total memory in GB.')
flags.DEFINE_integer('total_cpu', 32 * 5, 'cluster total CPU cores.')
flags.DEFINE_integer('min_shared_jobs', 3, 'resources can be shared by minimum how many jobs')


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
            # print logs
            logs_data = json.loads(flask.request.get_json())
            if 'fueling_zip_base64' in logs_data['job']:
                logs_data['job'].pop('fueling_zip_base64')
            logging.info(F'{job_id} submitted')
            logging.info(F'parameters: {json.dumps(logs_data)}')
            # Validate args
            if (arg.worker.count * arg.worker.memory * flags.FLAGS.min_shared_jobs
                    >= flags.FLAGS.total_memory):
                return json.dumps({'error': 'Too much memory requested!'}), HTTPStatus.BAD_REQUEST
            elif (arg.worker.count * arg.worker.cpu * flags.FLAGS.min_shared_jobs
                    >= flags.FLAGS.total_cpu):
                return json.dumps({'error': 'Too many cpu requested!'}), HTTPStatus.BAD_REQUEST

            if flags.FLAGS.debug:
                Utils.spark_submit(job_id, arg, False)
            else:
                threading.Thread(target=Utils.spark_submit, args=(job_id, arg)).start()
            return json.dumps({'job_id': job_id}), HTTPStatus.OK
        except json_format.ParseError as err:
            logging.error(err)
            return json.dumps({'error': 'Bad SparkSubmitArg format!'}), HTTPStatus.BAD_REQUEST


class OpenServiceSubmitJob(SparkSubmitJob):
    """submit open service jobs"""

    def post(self):
        """Accept user request, verify and process."""
        try:
            arg = json_format.Parse(flask.request.get_json(), SaasJobArg())
            logging.info('openservice submitted')
            logging.info(F'parameters: {flask.request.get_json()}')
            http_code, msg = JobProcessor(arg).process()
            logging.info(F'msg: {msg}')
        except Exception as err:
            logging.error(err)
            logging.error(traceback.format_exc())
            http_code = HTTPStatus.BAD_REQUEST
            msg = 'Wrong job argument'
        return msg, http_code


flask_app = flask.Flask(__name__)
api = flask_restful.Api(flask_app)
api.add_resource(SparkSubmitJob, '/')
api.add_resource(OpenServiceSubmitJob, '/open-service')


def main(argv):
    if flags.FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    flask_app.run(host='0.0.0.0', port=8000, debug=flags.FLAGS.debug)


if __name__ == '__main__':
    app.run(main)
