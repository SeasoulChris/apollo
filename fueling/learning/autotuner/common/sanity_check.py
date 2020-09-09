import requests
import re

from fueling.common.job_utils import JobUtils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

from fueling.learning.autotuner.proto.dynamic_model_info_pb2 import DynamicModel
from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs, TuningModule


class LocalJobUtils():
    def save_job_failure_code(self, error_code):
        logging.info(f'Failure code: {error_code}')

    def save_job_failure_detail(self, msg):
        logging.info(f'Failure detail: {msg}')


class AutotunerSanityCheck():
    valid_scenarios = set([
        # control autotune scenarios
        11014, 11015, 11016, 11017, 11018, 11019, 11020,
        30019, 30020, 30021, 30022, 30023, 30024, 30025,
        # public scenarios
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 18, 19, 20,
        21, 22, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 94, 95, 96, 97, 98, 99,
        101, 847, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908,
        10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009,
        10010, 10011, 10012, 10013, 10014, 10015, 10016, 10017, 10018, 10019,
        10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029,
        10030, 10031, 10032, 10033, 10034, 10035, 10036, 10038, 10039, 10041,
        10042, 10043, 10044, 10045, 10047, 10049, 10050, 10051, 10052, 10053,
        10055, 10056, 10057, 10058, 10059, 10060, 10061, 10062, 10064, 10065,
        10066, 10068, 10070, 10071, 10072, 10073, 10074, 10076, 10077, 10078,
        10079, 10080, 10081, 10082, 10083, 10084, 10085, 10086, 10087, 10088,
        10089, 10090, 10091, 10092, 10093, 10094, 10095, 10096, 10097, 10098,
        10099, 10100, 10872, 10873, 10876, 10877, 10880, 10887, 10896, 10948,
        10949, 10952, 10957, 10966,
    ])

    def __init__(self, is_local, job_id, input_path):
        self.job_utils = LocalJobUtils() if is_local else JobUtils(job_id)
        self.config_file_path = input_path
        self.config_pb = None
        self.repo = None

    def has_config_file(self):
        if not file_utils.file_exists(self.config_file_path):
            self.job_utils.save_job_failure_code('E103')
            self.job_utils.save_job_failure_detail(
                f'Config file {self.config_file_path} not found.')
            return False
        return True

    def is_config_file_readable(self):
        try:
            self.config_pb = TunerConfigs()
            proto_utils.get_pb_from_text_file(
                self.config_file_path, self.config_pb)
            return True
        except Exception as error:
            self.job_utils.save_job_failure_code('E100')
            self.job_utils.save_job_failure_detail(
                F'Cannot parse config file as binary or text proto: {error}')
            return False

    def check_required_fields(self):
        if not self.config_pb.git_info.commit_id:
            self.job_utils.save_job_failure_code('E101')
            self.job_utils.save_job_failure_detail(
                'missing git_info.commit_id')
            return False

        param = self.config_pb.tuner_parameters
        module_name = TuningModule.Name(param.user_tuning_module)
        if module_name != 'CONTROL':
            self.job_utils.save_job_failure_code('E102')
            self.job_utils.save_job_failure_detail(
                F'Bad tuning module "{module_name}" found')
            return False

        return True

    def check_n_iter(self):
        if self.config_pb.tuner_parameters.n_iter > 1000:
            self.job_utils.save_job_failure_code('E102')
            self.job_utils.save_job_failure_detail('n_iter must be <= 1000')
            return False

        return True

    def check_scenarios(self):
        invalid_scenarios = set(filter(
            lambda scenario: scenario not in AutotunerSanityCheck.valid_scenarios,
            self.config_pb.scenarios.id
        ))

        if invalid_scenarios:
            self.job_utils.save_job_failure_code('E102')
            self.job_utils.save_job_failure_detail(
                f'Invalid scenarios {invalid_scenarios} found')
            return False

        return True

    def get_git_file(self, file_path):
        try:
            owner, name, commit = self.repo['owner'], self.repo['name'], self.repo['commit']
            url = f'https://raw.githubusercontent.com/{owner}/{name}/{commit}/{file_path}'
            res = requests.get(url)
            if res.status_code != 200:
                return None
            else:
                return res
        except BaseException as ex:
            logging.error(ex)
            return None

    def check_git_file_exists(self, file_path):
        return self.get_git_file(file_path) is not None

    def check_git_repo(self):
        if not self.config_pb.git_info.repo:
            return True

        match = re.match('^https://github.com/([^//]+)/([^//]+).git$', self.config_pb.git_info.repo)
        if not match:
            self.job_utils.save_job_failure_code('E102')
            self.job_utils.save_job_failure_detail(
                'Invalid git_info.repo found. Please provide repo as HTTPS URL.')
            return False

        (owner, name) = match.groups()
        self.repo = {'owner': owner, 'name': name, 'commit': self.config_pb.git_info.commit_id}
        return True

    def check_apollo_env(self):
        if not self.config_pb.git_info.repo:
            return True

        def is_apollo_60_plus(env):
            # apollo 6.0+ has env.json file w/ OS set to ubuntu 18.04 and python 3.6
            logging.info(f'Apollo env: {env}')
            if env['OS']['distro'] != 'Ubuntu' or \
                    env['OS']['release'] != '18.04' or env['Python'] != '3.6':
                self.job_utils.save_job_failure_code('E102')
                self.job_utils.save_job_failure_detail(
                    f'Invalid apollo env found: {env}')
                return False
            return True

        res = self.get_git_file('env.json')
        if res:
            return is_apollo_60_plus(res.json())
        else:
            self.job_utils.save_job_failure_code('E102')
            self.job_utils.save_job_failure_detail('This service only supports apollo 6.0+')
            return False

    def check_control_conf(self):
        control_conf_path = self.config_pb.tuner_parameters.user_conf_filename
        if not control_conf_path.startswith('/apollo/modules'):
            self.job_utils.save_job_failure_code('E102')
            self.job_utils.save_job_failure_detail(
                'user_conf_filename is not defined under /apollo/modules')
            return False

        if not self.check_git_file_exists(control_conf_path[len('/apollo/'):]):
            self.job_utils.save_job_failure_code('E103')
            self.job_utils.save_job_failure_detail(
                f'{control_conf_path} not found from the given git repo.')
            return False

        return True

    def check_dynamic_model(self):
        try:
            model_name = DynamicModel.Name(self.config_pb.dynamic_model)
            if model_name == 'PERFECT_CONTROL':
                # PERFECT_CONTROL is the default value set by proto.
                # As this field isn't set, use OWN_MODEL instead.
                logging.info('No dynamic model specified. Set to OWN_MODEL')
                self.config_pb.dynamic_model = DynamicModel.OWN_MODEL
                model_name = 'OWN_MODEL'

            # Default repo is apollo's master, which can be used for other dynamic models
            if model_name == 'OWN_MODEL':
                if not self.config_pb.git_info.repo:
                    self.job_utils.save_job_failure_code('E101')
                    self.job_utils.save_job_failure_detail('missing git_info.repo')
                    return False

                if not self.check_git_file_exists('modules/control/conf/dynamic_model_forward.bin'):
                    self.job_utils.save_job_failure_code('E103')
                    self.job_utils.save_job_failure_detail('missing forward model')
                    return False

                if not self.check_git_file_exists(
                        'modules/control/conf/dynamic_model_backward.bin'):
                    self.job_utils.save_job_failure_code('E103')
                    self.job_utils.save_job_failure_detail('missing backward model')
                    return False

            return True
        except Exception as error:
            logging.error(error)
            self.job_utils.save_job_failure_code('E102')
            self.job_utils.save_job_failure_detail(
                f'Invalid dynamic model "{model_name}"')
            return False

    def check(self):
        check_list = [
            self.has_config_file,
            self.is_config_file_readable,
            self.check_required_fields,
            self.check_n_iter,
            self.check_scenarios,

            # NOTE: the following checks require access to user's
            # github repo so check_git_repo must happen first
            self.check_git_repo,
            self.check_apollo_env,
            self.check_control_conf,
            self.check_dynamic_model,
        ]

        for check_item in check_list:
            if not check_item():
                logging.info('Sanity check failed.')
                return False

        logging.info('Sanity check passed.')
        return True
