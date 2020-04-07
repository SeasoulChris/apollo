#!/usr/bin/env python3

import grpc

import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.learning.autotuner.proto.sim_service_pb2 as sim_service_pb2
import fueling.common.logging as logging


class CostComputationClient(object):
    """The Python implementation of the Cost Computation GRPC client."""

    CHANNEL_URL = "localhost:50052"

    @classmethod
    def set_channel(cls, channel):
        cls.CHANNEL_URL = channel

    @staticmethod
    def get_dynamic_model_enum(name):
        try:
            enum = sim_service_pb2.DynamicModel.Value(name.upper())
            return enum
        except Exception as error:
            raise ValueError(F"Dynamic model not found. \n\t{error}")

    @staticmethod
    def construct_request(commit_id, configs, scenario_ids, dynamic_model_name):
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
        dynamic_model = CostComputationClient.get_dynamic_model_enum(dynamic_model_name)

        # construct
        request = cost_service_pb2.Request()
        request.git_info.commit_id = commit_id
        request.dynamic_model = dynamic_model
        request.scenario_id.extend(scenario_ids)
        for (config_id, path_2_pb2) in configs.items():
            if not isinstance(path_2_pb2, dict):
                raise TypeError(
                    f"incorrect config-item type found: {path_2_pb2}.")

            for (path, config_pb2) in path_2_pb2.items():
                request.config[config_id].model_config[path] = config_pb2

        return request

    @classmethod
    def compute_mrac_cost(cls, commit_id, configs, scenario_ids, dynamic_model_name):
        try:
            with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
                stub = cost_service_pb2_grpc.CostComputationStub(channel)
                logging.info(
                    f"Sending compute request with commit_id {commit_id} ...")
                request = CostComputationClient.construct_request(
                    commit_id, configs, scenario_ids, dynamic_model_name)
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
