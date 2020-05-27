#!/usr/bin/env python
"""The Python implementation of the Simulation's GRPC client."""

import grpc

import fueling.learning.autotuner.proto.sim_service_pb2 as sim_service_pb2
import fueling.learning.autotuner.proto.git_info_pb2 as git_info_pb2
import fueling.learning.autotuner.proto.sim_service_pb2_grpc as sim_service_pb2_grpc
import fueling.common.logging as logging

REQUEST_TIMEOUT_IN_SEC = 15 * 60

# channel options: https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/grpc_types.h
CHANNEL_OPTIONS = [
    ('grpc.keepalive_timeout_ms', 60000),
    ('grpc.keepalive_time_ms', 5 * 60000),
    ('grpc.keepalive_permit_without_calls', 1),
    ('grpc.http2.max_pings_without_data', 12),
]

class SimClient(object):
    CHANNEL_URL = "localhost:50051"

    @classmethod
    def set_channel(cls, channel_url):
        cls.CHANNEL_URL = channel_url

    @classmethod
    def send_request(cls, request_name, request_payload):
        with grpc.insecure_channel(
                target=cls.CHANNEL_URL,
                compression=grpc.Compression.Gzip,
                options=CHANNEL_OPTIONS) as channel:

            stub = sim_service_pb2_grpc.SimServiceStub(channel)
            request_function = getattr(stub, request_name)
            status = request_function(request_payload, timeout=REQUEST_TIMEOUT_IN_SEC)

        return status

    @classmethod
    def initialize(cls, service_token, git_info, num_workers, dynamic_model):
        logging.info(f"Initializing service {service_token}  ...")
        init_param = sim_service_pb2.InitParam(
            git_info=git_info,
            num_workers=num_workers, token=service_token, dynamic_model=dynamic_model
        )

        return cls.send_request('Initialize', init_param)

    @classmethod
    def run_scenario(
        cls, service_token, iteration_id, scenario_id, config, record_output_path, record_output_file
    ):
        logging.info(f"Running scenario {scenario_id} for {record_output_file} ...")
        job_info = sim_service_pb2.JobInfo(
            token=service_token,
            scenario=scenario_id,
            model_config=config,
            record_output_path=record_output_path,
            record_output_file=record_output_file,
            iteration_id=iteration_id,
        )

        return cls.send_request('RunScenario', job_info)

    @classmethod
    def close(cls, service_token_str):
        logging.info(f"Tearing down service {service_token_str} ...")
        token = sim_service_pb2.Token(token=service_token_str)

        return cls.send_request('TearDown', token)
