#!/usr/bin/env python

import grpc

# import fueling.autotuner.grpc.sim_service_pb2 as sim_service_pb2
import fueling.autotuner.grpc.cost_computation_service_pb2 as cost_service_pb2
import fueling.autotuner.grpc.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.common.logging as logging

class CostComputationClient(object):
    """The Python implementation of the Cost Computation GRPC client."""

    CHANNEL_URL = "localhost:50052"

    @classmethod
    def compute_mrac_cost(cls, commit, configs):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = cost_service_pb2_grpc.CostComputationStub(channel)

            # Construct request
            request = cost_service_pb2.Request()
            request.git_info.commit = commit
            for single_config in configs:
                proto_config = cost_service_pb2.ModelConfig(model_config=single_config)
                request.config.append(proto_config)

            logging.info(f"Triggering compute with commit {commit} ...")
            status = stub.ComputeMracCost(request)

        if status.code == 0:
            logging.info(f"Done computing cost {status.score}")
            return status.score
        else:
            logging.error(f"Failed to compute cost: {status.message}")
            return float('nan')
