#!/usr/bin/env python3

import getpass
import grpc
import signal

from fueling.learning.autotuner.common.utils import run_with_retry
import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.learning.autotuner.proto.sim_service_pb2 as sim_service_pb2
import fueling.common.logging as logging

MAX_RUNNING_ROLE_LENGTH = 18
MAX_RETRIES = 3

KEEP_ALIVE_TIME_IN_SEC = 5 * 60
OPERATION_TIMEOUT_IN_SEC = {
    'Initialize': 30 * 60,
    # first time usually needs longer time to pull code, map, binary, etc
    'FirstComputeCost': 45 * 60,
    'ComputeCost': 10 * 60,
    'Close': 5 * 60,
    'Default': 10 * 60,
}


class CostComputationClient(object):
    """The Python implementation of the Cost Computation GRPC client."""

    def __init__(self, channel_url="localhost:50052",
                 commit_id=None, scenario_ids=None,
                 dynamic_model=None,
                 running_role_postfix=None):
        self.service_token = None
        self.max_retries = MAX_RETRIES
        self.channel, self.stub = self.create_channel_and_stub(channel_url)
        self.running_role_postfix = running_role_postfix
        self.first_cost_computation = True

        if commit_id and scenario_ids and dynamic_model:
            self.initialize(commit_id, scenario_ids, dynamic_model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self.channel.close()

    def create_channel_and_stub(self, channel_url):
        logging.info(f'Setting up grpc connection to {channel_url}')

        # channel options:
        # https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/grpc_types.h
        max_request_timeout_in_sec = max(OPERATION_TIMEOUT_IN_SEC.values())
        channel_options = [
            ('grpc.keepalive_timeout_ms', 60000),
            ('grpc.keepalive_time_ms', KEEP_ALIVE_TIME_IN_SEC * 1000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data',
             int(max_request_timeout_in_sec / KEEP_ALIVE_TIME_IN_SEC)),
        ]
        channel = grpc.insecure_channel(
            target=channel_url,
            compression=grpc.Compression.Gzip,
            options=channel_options)
        stub = cost_service_pb2_grpc.CostComputationStub(channel)

        return channel, stub

    def set_max_retries(self, retries):
        self.max_retries = retries

    def is_initialized(self):
        return self.service_token is not None

    def construct_init_request(self, commit_id, scenario_ids, dynamic_model, running_role):
        # validate inputs
        if not isinstance(scenario_ids, list):
            raise TypeError(
                f"incorrect scenario type found. Should be list, but found {type(scenario_ids)}."
            )
        if not scenario_ids:
            raise ValueError("Scenario list cannot be empty.")

        # construct request
        request = cost_service_pb2.InitRequest()
        request.git_info.commit_id = commit_id
        request.scenario_id.extend(scenario_ids)
        request.dynamic_model = dynamic_model
        request.running_role = running_role

        return request

    def construct_compute_request(self, configs, cost_conf_file):
        # validate inputs
        if not isinstance(configs, dict):
            raise TypeError(
                f"incorrect config type found. Should be dict, but found {type(configs)}."
            )

        # construct request
        request = cost_service_pb2.ComputeRequest()
        request.token = self.service_token
        if cost_conf_file:
            request.cost_computation_conf_filename = cost_conf_file
        for (config_id, path_2_pb2) in configs.items():
            if not isinstance(path_2_pb2, dict):
                raise TypeError(
                    f"incorrect config-item type found: {path_2_pb2}.")

            for (path, config_pb2) in path_2_pb2.items():
                request.config[config_id].model_config[path] = config_pb2

        return request

    def construct_close_request(self):
        return cost_service_pb2.CloseRequest(token=self.service_token)

    def send_request_with_retry(self, request_name, request_payload, request_timeout_sec=None):
        request_function = getattr(self.stub, request_name)

        default_timeout = OPERATION_TIMEOUT_IN_SEC['Default']
        timeout = request_timeout_sec or OPERATION_TIMEOUT_IN_SEC.get(request_name, default_timeout)
        logging.info(f"Running {request_name} with {timeout} sec timeout.")
        for retry in range(self.max_retries):
            try:
                if retry > 0:
                    logging.info(f'Retry {request_name} request. Retry count {retry} ...')

                future = request_function.future(request_payload, timeout=timeout)

                def cancel_request(unused_signum, unused_frame):
                    print(f'Cancelling {request_name} request ...')
                    future.cancel()
                    raise Exception('Request Cancelled.')

                signal.signal(signal.SIGINT, cancel_request)
                response = future.result()

            except grpc.RpcError as rpc_error:
                if retry == (self.max_retries - 1):
                    raise rpc_error
                else:
                    logging.error(rpc_error)
            else:
                break

        status = response.status
        if status.code != 0:
            raise Exception(f"failed to run {request_name}. "
                            f"\n\tcode: {status.code}"
                            f"\n\tmessage: {status.message}")

        return response

    def set_token(self, service_token):
        if self.is_initialized():
            self.close()

        self.service_token = service_token

    def initialize(self, commit_id, scenario_ids, dynamic_model):
        if self.is_initialized():
            logging.info(f"Service {self.service_token} has been initialized")
            return

        logging.info(f"Initializing service for commit {commit_id} with training scenarios "
                     f"{scenario_ids} ...")

        running_role = getpass.getuser()
        if self.running_role_postfix:
            running_role += f"-{self.running_role_postfix}"

        request = self.construct_init_request(
            commit_id, scenario_ids, dynamic_model, running_role[:MAX_RUNNING_ROLE_LENGTH])
        response = self.send_request_with_retry('Initialize', request)
        self.service_token = response.token
        logging.info(f"Service {self.service_token} initialized ")

    def compute_cost(self, configs, cost_config_file=None):
        if not self.is_initialized():
            logging.error("Please initialize first.")
            return None

        logging.info(f"Sending compute request to service {self.service_token} ...")

        request_payload = self.construct_compute_request(configs, cost_config_file)
        request_timeout = \
            OPERATION_TIMEOUT_IN_SEC['FirstComputeCost'] if self.first_cost_computation else None
        self.first_cost_computation = False

        response = run_with_retry(
            self.max_retries,
            self.send_request_with_retry,
            'ComputeCost',
            request_payload,
            request_timeout)
        logging.info(f"Service {self.service_token} finished computing cost {response.score} for "
                     f"iteration {response.iteration_id}")
        return response.iteration_id, response.score

    def close(self):
        if not self.is_initialized():
            return

        logging.info(f"Closing {self.service_token} service ...")
        request = self.construct_close_request()
        response = self.send_request_with_retry('Close', request)
        logging.info(f"Service {self.service_token} closed.")
        self.service_token = None
