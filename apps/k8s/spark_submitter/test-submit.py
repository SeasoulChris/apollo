#!/usr/bin/env python

from absl import flags
from absl import logging
import requests

# Guideline for resource allocation:
# 1. Maximum resource of an EXECUTOR is the resource of a node, i.e.:
#    CPU Cores: 32
#    Memory: 256GB
#    Ephemeral Storage: 5TB
# 2. Maximum resource of a JOB is the total resource of all 5 nodes, i.e.:
#    CPU Cores: 160
#    Memory: 1.25TB
#    Ephemeral Storage: 25TB
# Show current cluster resource usage with "kubectl top nodes".

# User.
flags.DEFINE_string('submitter', None, 'Job submitter.')
flags.DEFINE_string('running_role', None, 'Job running role.')

# Env.
flags.DEFINE_string('docker_image', 'hub.baidubce.com/apollo/spark:latest', 'Docker image.')
flags.DEFINE_string('conda_env', 'fuel-py36', 'Conda env name.')
flags.DEFINE_string('node_selector', 'CPU', 'Node selector.')
flags.DEFINE_string('log_verbosity', 'INFO', 'Log verbosity.')

# Job.
flags.DEFINE_string('entrypoint', None, 'Job entrypoint.')
flags.DEFINE_string('fueling_zip_path', None, 'Fueling zip path.')

# Worker.
flags.DEFINE_integer('worker_count', 1, 'Worker count.')
flags.DEFINE_integer('worker_cpu', 1, 'Worker CPU cores.')
flags.DEFINE_integer('worker_memory', 12, 'Worker memory in GB.')
flags.DEFINE_integer('worker_disk', 20, 'Worker disk in GB.')

# Partner.
flags.DEFINE_boolean('partner_storage_writable', False, 'Mount partner storage as writable.')
flags.DEFINE_string('partner_bos_bucket', None, 'Partner bos bucket.')
flags.DEFINE_string('partner_bos_region', None, 'Partner bos region.')
flags.DEFINE_string('partner_bos_access', None, 'Partner bos access.')
flags.DEFINE_string('partner_bos_secret', None, 'Partner bos secret.')

# requests.post json.
