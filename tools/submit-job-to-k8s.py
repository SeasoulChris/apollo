#!/usr/bin/env python3

import base64
import getpass
import glob
import io
import json
import os
import sys
import time
import zipfile

from absl import app
from absl import flags
from absl import logging
import requests


# User.
flags.DEFINE_string('running_role', None, 'Running as another role instead of the job submitter.')

# Env.
flags.DEFINE_string('image', 'hub.baidubce.com/apollo/spark:latest', 'Docker image.')
flags.DEFINE_string('env', 'fuel-py27-cyber', 'Conda env name.')
flags.DEFINE_enum('node_selector', 'CPU', ['CPU', 'GPU', 'ANY'], 'Node selector.')
flags.DEFINE_enum('log_verbosity', 'INFO', ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                  'Log verbosity.')

# Job.
flags.DEFINE_string('main', None, 'Job entrypoint.')
flags.DEFINE_string('fueling_zip_path', None, 'Fueling zip path.')
flags.DEFINE_string('flags', None, 'Job flags.')
flags.DEFINE_boolean('with_learning_algorithms', False,
                     'Whether to package the learning_algorithms folder.')
flags.DEFINE_boolean('wait', False, 'Whether to wait to finish.')

# Worker.
flags.DEFINE_integer('workers', 1, 'Worker count.')
flags.DEFINE_integer('cpu', 1, 'Worker CPU cores.')
flags.DEFINE_integer('memory', 12, 'Worker memory in GB.')
flags.DEFINE_integer('disk', 20, 'Worker disk in GB.')

# Partner.
flags.DEFINE_boolean('partner_storage_writable', False, 'Mount partner storage as writable.')

flags.DEFINE_string('partner_bos_bucket', None, 'Partner bos bucket.')
flags.DEFINE_string('partner_bos_region', None, 'Partner bos region.')
flags.DEFINE_string('partner_bos_access', None, 'Partner bos access.')
flags.DEFINE_string('partner_bos_secret', None, 'Partner bos secret.')

flags.DEFINE_string('partner_azure_storage_account', None, 'Partner Azure storage account.')
flags.DEFINE_string('partner_azure_storage_access_key', None, 'Partner Azure storage access key.')
flags.DEFINE_string('partner_azure_blob_container', None, 'Partner Azure blob container.')


def get_user():
    return {
        'submitter': getpass.getuser(),
        'running_role': flags.FLAGS.running_role or getpass.getuser(),
    }


def get_env():
    return {
        'conda_env': flags.FLAGS.env,
        'docker_image': flags.FLAGS.image,
        'node_selector': flags.FLAGS.node_selector,
        'log_verbosity': flags.FLAGS.log_verbosity,
    }


def get_job():
    job = {
        'entrypoint': flags.FLAGS.main,
        'flags': flags.FLAGS.flags,
    }
    if flags.FLAGS.fueling_zip_path:
        # Use pre-packaged code in cloud.
        job['fueling_zip_path'] = flags.FLAGS.fueling_zip_path
    else:
        # Use local code.
        with io.BytesIO() as in_mem_zip:
            # Write in_mem_zip.
            with zipfile.ZipFile(in_mem_zip, 'w', zipfile.ZIP_DEFLATED) as fueling_zip:
                for f in glob.glob('fueling/**', recursive=True):
                    if os.path.isfile(f):
                        fueling_zip.write(f)
                if flags.FLAGS.with_learning_algorithms:
                    for f in glob.glob('learning_algorithms/**', recursive=True):
                        if os.path.isfile(f):
                            fueling_zip.write(f)

            fueling_zip = in_mem_zip.getvalue()
            logging.info('fueling.zip has %.2fMB.' % (len(fueling_zip) / (2**20)))
            job['fueling_zip_base64'] = base64.b64encode(fueling_zip).decode('ascii')
    return job


def get_worker():
    return {
        'count': flags.FLAGS.workers,
        'cpu': flags.FLAGS.cpu,
        'memory': flags.FLAGS.memory,
        'disk': flags.FLAGS.disk,
    }


def get_partner():
    partner = {'partner_storage_writable': flags.FLAGS.partner_storage_writable}
    if flags.FLAGS.partner_bos_bucket:
        partner['bos'] = {
            'bucket': flags.FLAGS.partner_bos_bucket,
            'access_key': flags.FLAGS.partner_bos_access,
            'secret_key': flags.FLAGS.partner_bos_secret,
            'region': flags.FLAGS.partner_bos_region,
        }
    elif flags.FLAGS.partner_azure_storage_account:
        partner['blob'] = {
            'storage_account': flags.FLAGS.partner_azure_storage_account,
            'storage_access_key': flags.FLAGS.partner_azure_storage_access_key,
            'blob_container': flags.FLAGS.partner_azure_blob_container,
        }


def main(argv):
    """Tool entrypoint."""
    if not os.path.exists('./fueling/'):
        logging.fatal('Must run from the apollo fule root directory.')
        sys.exit(1)

    # Construct argument according to apps/k8s/spark_submitter/spark_submit_arg.proto
    arg = {
        'user': get_user(),
        'env': get_env(),
        'job': get_job(),
        'worker': get_worker(),
        'partner': get_partner(),
    }

    # Submit job.

    # For debug purpose, you may use the following config.
    # KUBE_PROXY_HOST = 'localhost'  # If you use local kube proxy.
    # SUBMITTER = 'http://localhost:8000/'  # If you use local submitter.

    KUBE_PROXY_HOST = 'usa-data.baidu.com'
    KUBE_PROXY = 'http://{}:8001'.format(KUBE_PROXY_HOST)
    SERVICE = 'http:spark-submitter-service:8000'
    SUBMITTER = '{}/api/v1/namespaces/default/services/{}/proxy/'.format(KUBE_PROXY, SERVICE)

    if os.system('ping -c 1 {} > /dev/null 2>&1'.format(KUBE_PROXY_HOST)) != 0:
        logging.fatal('Cannot reach k8s proxy {}. Are you running in intranet?'.format(KUBE_PROXY_HOST))
        sys.exit(1)
    res = requests.post(SUBMITTER, json=json.dumps(arg))
    payload = json.loads(res.json() or '{}')

    # Process result.
    if not res.ok:
        logging.error('Failed to submit job: HTTP {}, {}'.format(
            res.status_code, payload.get('error')))
        return

    job_id = payload.get('job_id')
    logging.info('Job {} submitted!'.format(job_id))
    # TODO: logging.info('View your task at ...')

    # Wait until job finishes.
    if not flags.FLAGS.wait:
        return
    WAIT_INTERVAL_SECONDS = 3
    END_STATUS = {'Completed', 'Error'}
    job_status = None
    while job_status not in END_STATUS:
        time.sleep(WAIT_INTERVAL_SECONDS)
        res = requests.get(SUBMITTER, params={'job_id': job_id})
        if res.ok:
            job_status = json.loads(res.json() or '{}').get('status')
            logging.info('Job is {}...'.format(job_status))
        else:
            logging.error('Failed to get job status.')
    if job_status == 'Error':
        sys.exit(1)


if __name__ == '__main__':
    app.run(main)
