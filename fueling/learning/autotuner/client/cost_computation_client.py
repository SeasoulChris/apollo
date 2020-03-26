#!/usr/bin/env python3

import grpc

import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.common.logging as logging


class CostComputationClient(object):
    """The Python implementation of the Cost Computation GRPC client."""

    CHANNEL_URL = "localhost:50052"

    @classmethod
    def set_channel(cls, channel):
        cls.CHANNEL_URL = channel

    @staticmethod
    def construct_request(commit_id, configs, scenario_ids):
        # validate inputs
        if not isinstance(configs, dict):
            raise TypeError(
                f"incorrect config type found. Should be dict, but found {type(configs)}."
            )
        if not isinstance(scenario_ids, list):
            raise TypeError(
                f"incorrect scenario type found. Should be list, but found {type(scenario_ids)}."
            )
        if not scenario_ids:
            raise ValueError("Scenario list cannot be empty.")

        # construct
        request = cost_service_pb2.Request()
        request.git_info.commit_id = commit_id
        request.scenario_id.extend(scenario_ids)
        for (config_id, path_2_pb2) in configs.items():
            if not isinstance(path_2_pb2, dict):
                raise TypeError(
                    f"incorrect config-item type found: {path_2_pb2}.")

            for (path, config_pb2) in path_2_pb2.items():
                request.config[config_id].model_config[path] = config_pb2

        return request

    @classmethod
    def compute_mrac_cost(cls, commit_id, configs, scenario_ids):
        try:
            with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
                stub = cost_service_pb2_grpc.CostComputationStub(channel)

                logging.info(
                    f"Sending compute request with commit_id {commit_id} ...")
                request = CostComputationClient.construct_request(
                    commit_id, configs, scenario_ids)
                response = stub.ComputeMracCost(request)

            status = response.status
            if status.code == 0:
                logging.info(
                    f"Done computing cost {response.score} for training_id {response.training_id}")
                return response.training_id, response.score
            else:
                logging.error(f"Error: {status.message}")
                return None

        except Exception as error:
            logging.error(f"Error: {error}")
            return None
