#!/usr/bin/env python
"""The Python implementation of the Simulation's GRPC client."""

import grpc
import time

import fueling.learning.autotuner.proto.sim_service_pb2 as sim_service_pb2
import fueling.learning.autotuner.proto.git_info_pb2 as git_info_pb2
import fueling.learning.autotuner.proto.sim_service_pb2_grpc as sim_service_pb2_grpc
import fueling.common.logging as logging


class SimClient(object):
    CHANNEL_URL = "localhost:50051"

    @classmethod
    def set_channel(cls, channel_url):
        cls.CHANNEL_URL = channel_url

    @classmethod
    def initialize(cls, service_token, git_info, num_workers, dynamic_model):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.SimServiceStub(channel)
            init_param = sim_service_pb2.InitParam(
                git_info=git_info,
                num_workers=num_workers, token=service_token, dynamic_model=dynamic_model
            )

            logging.info(f"Initializing service {service_token}  ...")
            status = stub.Initialize(init_param)
        return status

    @classmethod
    def run_scenario(
        cls, service_token, iteration_id, scenario_id, config, record_output_path, record_output_file
    ):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.SimServiceStub(channel)

            job_info = sim_service_pb2.JobInfo(
                token=service_token,
                scenario=scenario_id,
                model_config=config,
                record_output_path=record_output_path,
                record_output_file=record_output_file,
                iteration_id=iteration_id,
            )

            logging.info(f"Running scenario {scenario_id} for {record_output_file} ...")
            status = stub.RunScenario(job_info)
        return status

    @classmethod
    def close(cls, service_token_str):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.SimServiceStub(channel)
            token = sim_service_pb2.Token(token=service_token_str)

            logging.info(f"Tearing down service {service_token_str} ...")
            status = stub.TearDown(token)
        return status
