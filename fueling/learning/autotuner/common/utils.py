import re
import requests
import time

from fueling.learning.autotuner.proto.dynamic_model_info_pb2 import DynamicModel
import fueling.common.logging as logging

DEFAULT_SCENARIOS = [11014, 11015, 11016, 11017, 11018, 11019, 11020]


def run_with_retry(max_retries, func, *params):
    """A wrapper to run a given function with certain amount of retries"""
    for trial in range(max_retries):
        try:
            if trial > 0:
                logging.info(f'Retry function {func.__name__}. Retry count {trial} ...')
            result = func(*params)
            return result
        except Exception as error:
            if trial == (max_retries - 1):
                raise error
            else:
                logging.error(error)
                time.sleep(10)


def set_tuner_config_default(tuner_param_config_pb):
    # default scenarios
    if not tuner_param_config_pb.scenarios.id:
        tuner_param_config_pb.scenarios.id[:] = DEFAULT_SCENARIOS
    logging.info(f"Training scenarios are {tuner_param_config_pb.scenarios.id}")

    # default dynamic model
    model_name = DynamicModel.Name(tuner_param_config_pb.dynamic_model)
    if model_name == 'PERFECT_CONTROL':
        # PERFECT_CONTROL is the default value set by proto.
        # As this field isn't set, use OWN_MODEL instead.
        logging.info('No dynamic model specified, set to OWN_MODEL')
        tuner_param_config_pb.dynamic_model = DynamicModel.OWN_MODEL

    git_info = tuner_param_config_pb.git_info
    # default repo
    if not git_info.repo:
        git_info.repo = 'https://github.com/ApolloAuto/apollo.git'

    # default commit id
    if not git_info.commit_id:
        match = re.match('^https://github.com/([^//]+)/([^//]+).git$', git_info.repo)
        if match:
            (owner, name) = match.groups()
            url = f'https://api.github.com/repos/{owner}/{name}/git/refs/heads/master'
            logging.info(f'Searching latest commit ID for {owner}\'s {name}')
            try:
                res = requests.get(url)
                if res.status_code == 200:
                    git_info.commit_id = res.json()['object']['sha']
                    logging.info(f'Set git_info.commit_id to {git_info.commit_id}')
            except BaseException as ex:
                logging.error(f'Cannot get head commit id: {ex}')
