#!/usr/bin/env python
"""The Python implementation of the Simulation's GRPC client."""

import grpc

import fueling.autotuner.proto.sim_service_pb2 as sim_service_pb2
import fueling.autotuner.proto.git_info_pb2 as git_info_pb2
import fueling.autotuner.proto.sim_service_pb2_grpc as sim_service_pb2_grpc
import fueling.common.logging as logging


class SimClient(object):
    CHANNEL_URL = "localhost:50051"

    @classmethod
    def set_channel(cls, channel_url):
        cls.CHANNEL_URL = channel_url

    @classmethod
    def trigger_build(cls, commit_id):
        try:
            with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
                stub = sim_service_pb2_grpc.SimServiceStub(channel)
                git_info = git_info_pb2.GitInfo(commit_id=commit_id)

                logging.info(f"Triggering build with commit_id {commit_id} ...")
                status = stub.TriggerBuildJob(git_info)

            if status.code == 0:
                logging.info(f"Done building apollo.")
                return True
            else:
                logging.error(f"Failed to build apollo: {status.message}")
                return False
        except Exception as error:
            logging.error(f"SimClient {cls.CHANNEL_URL} exception: {error}")
            return False

    @classmethod
    def run_scenario(
        cls, training_id, commit_id, scenario, config, record_output_path, record_output_filename
    ):
        try:
            with grpc.insecure_channel(cls.CHANNEL_URL) as channel:
                stub = sim_service_pb2_grpc.SimServiceStub(channel)

                git_info = git_info_pb2.GitInfo(commit_id=commit_id)
                job_info = sim_service_pb2.JobInfo(
                    git_info=git_info,
                    scenario=scenario,
                    model_config=config,
                    record_output_path=record_output_path,
                    record_output_filename=record_output_filename,
                    training_id=training_id,
                )

                logging.info(f"Running scenario {scenario} for {record_output_filename} ...")
                status = stub.RunScenario(job_info)

            if status.code == 0:
                logging.info(f"Done running scenario {scenario} for {record_output_filename}.")
                return True
            else:
                logging.error(f"Failed to run scenario {scenario} for {record_output_filename}: {status.message}")
                return False
        except Exception as error:
            logging.error(f"SimClient {cls.CHANNEL_URL} exception: {error}")
            return False
