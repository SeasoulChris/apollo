#!/usr/bin/env python3

import base64
import getpass
import json
import os
import pprint
import sys
import time

try:
    from absl import app
    from absl import flags
    from absl import logging
except:
    print('Cannot import absl, you may need to run "sudo python3 -m pip install absl-py".')

try:
    import requests
except:
    print('Cannot import requests, you may need to run "sudo python3 -m pip install requests".')


# User.
flags.DEFINE_string('role', 'apollo', 'Running as another role instead of the job submitter.')

# Env.
flags.DEFINE_string('image', 'hub.baidubce.com/apollo/spark:bazel2', 'Docker image.',
                    short_name='i')
flags.DEFINE_enum('node_selector', 'CPU', ['CPU', 'GPU', 'ANY'], 'Node selector.')
flags.DEFINE_enum('log_verbosity', 'INFO', ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                  'Log verbosity.')

# Job.
flags.DEFINE_string('main', None, 'Pre-built py_binary zip file.')
flags.DEFINE_string('flags', None, 'Job flags.')
flags.DEFINE_boolean('wait', False, 'Whether to wait to finish.')

# Worker.
flags.DEFINE_integer('workers', 1, 'Worker count.', short_name='w')
flags.DEFINE_integer('cpu', 1, 'Worker CPU cores.', short_name='c')
flags.DEFINE_integer('memory', 12, 'Worker memory in GB.', short_name='m')
flags.DEFINE_integer('disk', 20, 'Worker disk in GB.', short_name='d')

# Partner.
flags.DEFINE_boolean('partner_storage_writable', False, 'Mount partner storage as writable.')

flags.DEFINE_string('partner_bos_bucket', None, 'Partner bos bucket.')
flags.DEFINE_string('partner_bos_region', None, 'Partner bos region.')
flags.DEFINE_string('partner_bos_access', None, 'Partner bos access.')
flags.DEFINE_string('partner_bos_secret', None, 'Partner bos secret.')


def get_user():
    return {
        'submitter': getpass.getuser(),
        'running_role': flags.FLAGS.role,
    }


def get_env():
    return {
        'docker_image': flags.FLAGS.image,
        'node_selector': flags.FLAGS.node_selector,
        'log_verbosity': flags.FLAGS.log_verbosity,
    }


def get_job():
    if not flags.FLAGS.main.endswith('.zip'):
        raise Exception('Please provide a built zip file to --main.')

    with open(flags.FLAGS.main, "rb") as fin:
        encoded_zip = base64.b64encode(fin.read()).decode('ascii')
    logging.info('Job has %.2fMB.' % (len(encoded_zip) / (2**20)))
    return {
        'entrypoint': flags.FLAGS.main,
        'fueling_zip_base64': encoded_zip,
        'flags': flags.FLAGS.flags,
    }


def get_worker():
    return {
        'count': flags.FLAGS.workers,
        'cpu': flags.FLAGS.cpu,
        'memory': flags.FLAGS.memory,
        'disk': flags.FLAGS.disk,
    }


def get_partner():
    partner = {'storage_writable': flags.FLAGS.partner_storage_writable}
    if flags.FLAGS.partner_bos_bucket:
        partner['bos'] = {
            'bucket': flags.FLAGS.partner_bos_bucket,
            'access_key': flags.FLAGS.partner_bos_access,
            'secret_key': flags.FLAGS.partner_bos_secret,
            'region': flags.FLAGS.partner_bos_region,
        }
    return partner


def main(argv):
    """Tool entrypoint."""
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

    arg['job']['fueling_zip_base64'] = ''
    logging.info('SparkSubmitArg is')
    pprint.PrettyPrinter(indent=2).pprint(arg)

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
    END_STATUS = {'Completed', 'Error', 'UnexpectedAdmissionError'}
    job_status = None
    while job_status not in END_STATUS:
        time.sleep(WAIT_INTERVAL_SECONDS)
        res = requests.get(SUBMITTER, params={'job_id': job_id})
        if res.ok:
            job_status = json.loads(res.json() or '{}').get('status')
            logging.info('Job is {}...'.format(job_status))
        else:
            logging.error('Failed to get job status.')
    if job_status != 'Completed':
        sys.exit(1)


if __name__ == '__main__':
    app.run(main)
