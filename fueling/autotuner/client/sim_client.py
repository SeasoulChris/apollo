#!/usr/bin/env python
"""The Python implementation of the Simulation's GRPC client."""

import grpc

import fueling.common.logging as logging

import fueling.autotuner.client.sim_service_pb2 as sim_service_pb2
import fueling.autotuner.client.sim_service_pb2_grpc as sim_service_pb2_grpc


class SimClient(object):
    CHANNEL_URL = "localhost:50051"

    @classmethod
    def trigger_build(cls, commit):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.AutoTunerStub(channel)
            git_info = sim_service_pb2.GitInfo(commit=commit)
            logging.info(f"Triggering build with commit {commit} ...")

            status = stub.TriggerBuildJob(git_info)

        logging.info(f"Finish building: {status}")
        return True

    @classmethod
    def run_scenario(
        cls, training_id, commit, scenario, config, output_path, output_file
    ):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.AutoTunerStub(channel)

            git_info = sim_service_pb2.GitInfo(commit=commit)
            job_info = sim_service_pb2.JobInfo(
                git_info=git_info,
                scenario=scenario,
                model_config=config,
                output_path=output_path,
                output_file=output_file,
                training_id=training_id,
            )
            status = stub.RunScenario(job_info)

        logging.info(f"Finish running scenario: {status.message}")
        return True
