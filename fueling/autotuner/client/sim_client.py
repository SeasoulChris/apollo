#!/usr/bin/env python
"""The Python implementation of the Simulation's GRPC client."""

import grpc

import fueling.common.logging as logging

import modules.data.fuel.fueling.autotuner.proto.sim_service_pb2 as sim_service_pb2
import modules.data.fuel.fueling.autotuner.proto.sim_service_pb2_grpc as sim_service_pb2_grpc


class SimClient(object):
    CHANNEL_URL = "localhost:50051"

    @classmethod
    def trigger_build(cls, commit_id):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.SimServiceStub(channel)
            git_info = sim_service_pb2.GitInfo(commit_id=commit_id)

            logging.info(f"Triggering build with commit_id {commit_id} ...")
            status = stub.TriggerBuildJob(git_info)

        if status.code == 0:
            logging.info(f"Done building apollo.")
            return True
        else:
            logging.error(f"Failed to build apollo: {status.message}")
            return False

    @classmethod
    def run_scenario(
        cls, training_id, commit_id, scenario, config, record_output_path, output_record_filename
    ):
        with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
            stub = sim_service_pb2_grpc.SimServiceStub(channel)

            git_info = sim_service_pb2.GitInfo(commit_id=commit_id)
            job_info = sim_service_pb2.JobInfo(
                git_info=git_info,
                scenario=scenario,
                model_config=config,
                record_output_path=record_output_path,
                output_record_filename=output_record_filename,
                training_id=training_id,
            )

            logging.info(f"Running scenario {scenario} for {output_record_filename} ...")
            status = stub.RunScenario(job_info)

        if status.code == 0:
            logging.info(f"Done running scenario {scenario} for {output_record_filename}.")
            return True
        else:
            logging.error(f"Failed to run scenario {scenario} for {output_record_filename}: {status.message}")
            return False
