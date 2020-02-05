#!/usr/bin/env python3

import grpc

import modules.data.fuel.fueling.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import modules.data.fuel.fueling.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc

import fueling.common.logging as logging


class CostComputationClient(object):
    """The Python implementation of the Cost Computation GRPC client."""

    CHANNEL_URL = "localhost:50052"

    @classmethod
    def compute_mrac_cost(cls, commit_id, configs):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = cost_service_pb2_grpc.CostComputationStub(channel)

            # Construct request
            request = cost_service_pb2.Request()
            request.git_info.commit_id = commit_id
            for single_config in configs:
                proto_config = cost_service_pb2.ModelConfig(model_config=single_config)
                request.config.append(proto_config)

            logging.info(f"Triggering compute with commit_id {commit_id} ...")
            response = stub.ComputeMracCost(request)

        status = response.status
        if status.code == 0:
            logging.info(f"Done computing cost {response.score}")
            return response.score
        else:
            logging.error(f"Failed to compute cost: {status.message}")
            return None
