#!/usr/bin/env python3

import grpc
import signal

import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.learning.autotuner.proto.sim_service_pb2 as sim_service_pb2
import fueling.common.logging as logging


class CostComputationClient(object):
    """The Python implementation of the Cost Computation GRPC client."""

    CHANNEL_URL = "localhost:50052"

    def __init__(self, commit_id=None, scenario_ids=None, dynamic_model=None):
        self.service_token = None
        if commit_id and scenario_ids and dynamic_model:
            self.initialize(commit_id, scenario_ids, dynamic_model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def set_channel(cls, channel):
        cls.CHANNEL_URL = channel

    def is_initialized(self):
        return self.service_token is not None

    def construct_init_request(self, commit_id, scenario_ids, dynamic_model):
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

    def send_request(self, request_name, request_payload):
        with grpc.insecure_channel(CostComputationClient.CHANNEL_URL, compression=grpc.Compression.Gzip) as channel:
            stub = cost_service_pb2_grpc.CostComputationStub(channel)
            request_function = getattr(stub, request_name)
            future = request_function.future(request_payload)

            def cancel_request(unused_signum, unused_frame):
                print(f'Cancelling {request_name} request...')
                future.cancel()
                raise Exception('Request Cancelled.')

            signal.signal(signal.SIGINT, cancel_request)
            response = future.result()

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

        logging.info(f"Initializing service for commit {commit_id} with training scenarios {scenario_ids} ...")
        request = self.construct_init_request(commit_id, scenario_ids, dynamic_model)
        response = self.send_request('Initialize', request)
        self.service_token = response.token
        logging.info(f"Service {self.service_token} initialized ")

    def compute_cost(self, configs, cost_config_file=None):
        if not self.is_initialized():
            logging.error("Please initialize first.")
            return None

        logging.info(f"Sending compute request to service {self.service_token} ...")
        request = self.construct_compute_request(configs, cost_config_file)
        response = self.send_request('ComputeCost', request)
        logging.info(f"Service {self.service_token} finished computing cost {response.score} for iteration {response.iteration_id}")
        return response.iteration_id, response.score

    def close(self):
        if not self.is_initialized():
            return

        logging.info(f"Closing {self.service_token} service ...")
        request = self.construct_close_request()
        response = self.send_request('Close', request)
        logging.info(f"Service {self.service_token} closed.")
        self.service_token = None
