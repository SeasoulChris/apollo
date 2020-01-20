#!/usr/bin/env python
"""The Python implementation of the Simulation's GRPC client."""

import grpc

import fueling.common.logging as logging

import fueling.autotuner.grpc.sim_service_pb2 as sim_service_pb2
import fueling.autotuner.grpc.sim_service_pb2_grpc as sim_service_pb2_grpc


class SimClient(object):
    CHANNEL_URL = "localhost:50051"

    @classmethod
    def trigger_build(cls, commit):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.AutoTunerStub(channel)
            git_info = sim_service_pb2.GitInfo(commit=commit)

            logging.info(f"Triggering build with commit {commit} ...")
            status = stub.TriggerBuildJob(git_info)

        if status.code == 0:
            logging.info(f"Done building apollo.")
            return True
        else:
            logging.error(f"Failed to build apollo: {status.message}")
            return False

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

            logging.info(f"Running scenario {scenario} for {output_file} ...")
            status = stub.RunScenario(job_info)

        if status.code == 0:
            logging.info(f"Done running scenario {scenario} for {output_file}.")
            return True
        else:
            logging.error(f"Failed to run scenario {scenario} for {output_file}: {status.message}")
            return False
