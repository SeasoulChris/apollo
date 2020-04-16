#!/usr/bin/env python

# standard packages
from concurrent import futures
from datetime import datetime
import json
import os
import time
import uuid

# third party packages
from absl import app
from absl import flags
import grpc

from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.learning.autotuner.client.sim_client import SimClient
import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

ONE_DAY_IN_SECONDS = 60 * 60 * 24

# Flags
flags.DEFINE_string(
    "sim_service_url", "localhost:50051", "channel url to sim service"
)

SERVER_PORT = "[::]:50052"

TMP_ROOT_DIR = "/mnt/bos/autotuner"


class CostComputation(cost_service_pb2_grpc.CostComputationServicer):
    """The Python implementation of the GRPC cost computation server."""

    def __init__(self):
        mode = "CLOUD" if flags.FLAGS.cloud else "LOCAL"
        logging.info(f"Running server in {mode} mode.")

        if flags.FLAGS.cloud:
            self.submit_job = CostComputation.SubmitJobToK8s
        else:
            self.submit_job = CostComputation.SubmitJobAtLocal

    @staticmethod
    def create_init_response(exit_code, message="", token=None):
        response = cost_service_pb2.InitResponse(token=token)
        response.status.code = exit_code
        response.status.message = message
        return response

    @staticmethod
    def create_compute_response(exit_code, message="", iteration_id="error", score=None):
        response = cost_service_pb2.ComputeResponse()
        response.status.code = exit_code
        response.status.message = message
        response.iteration_id = iteration_id
        if score is not None:
            response.score = score
        return response

    @staticmethod
    def SubmitJobAtLocal(options):
        job_cmd = "bazel run //fueling/learning/autotuner/cost_computation:control_cost_computation"
        option_strings = [f"--{name}={value}" for (name, value) in options.items()]
        cmd = f"cd /fuel; {job_cmd} -- {' '.join(option_strings)}"
        logging.info(f"Executing '{cmd}'")

        # TODO: exit_code does not work so far, check abseil's app to see how to set exit code
        exit_code = os.system(cmd)
        return os.WEXITSTATUS(exit_code) == 0

    @staticmethod
    def SubmitJobToK8s(options):
        entrypoint = "fueling/learning/autotuner/cost_computation/control_cost_computation.py"
        client = SparkSubmitterClient(entrypoint, {}, options)
        client.submit()
        return True

    def get_service_dir(self, token):
        return f"{TMP_ROOT_DIR}/{token}"

    def Initialize(self, request, context):
        if not request.git_info.commit_id:
            return CostComputation.create_init_response(1, "Commit ID not specified.")
        if not request.scenario_id:
            return CostComputation.create_init_response(1, "Scenario(s) not specified.")

        # Save config to a local file
        service_token = f"autotuner-{uuid.uuid4().hex}"
        tmp_dir = self.get_service_dir(service_token)
        file_utils.makedirs(tmp_dir)
        proto_utils.write_pb_to_text_file(request, f"{tmp_dir}/init_request.pb.txt")

        # init
        num_workers = len(request.scenario_id)
        SimClient.set_channel(flags.FLAGS.sim_service_url)
        status = SimClient.initialize(
            service_token,
            request.git_info,
            num_workers,
            request.dynamic_model)

        return CostComputation.create_init_response(
            status.code, status.message, service_token,
        )

    def ComputeCost(self, request, context):
        if not request.token:
            return CostComputation.create_compute_response(
                exit_code=1, message="Service token not specified.")

        iteration_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        tmp_dir = f"{self.get_service_dir(request.token)}/{iteration_id}"
        file_utils.makedirs(tmp_dir)

        # Save config to a local file
        proto_utils.write_pb_to_text_file(request, f"{tmp_dir}/compute_request.pb.txt")

        # submit job
        options = {
            "sim_service_url": flags.FLAGS.sim_service_url,
            "token": request.token,
            "iteration_id": iteration_id,
        }
        if request.cost_computation_conf_filename:
            options['cost_computation_conf_filename'] = request.cost_computation_conf_filename

        if not self.submit_job(options):
            return CostComputation.create_compute_response(
                exit_code=1, message="failed to compute cost."
            )

        # read and return score
        try:
            with open(f"{tmp_dir}/scores.out") as score_file:
                scores = json.loads(score_file.readline())

            response = CostComputation.create_compute_response(
                exit_code=0, message="Done.", iteration_id=iteration_id)
            for (config_id, weighted_score) in scores.items():
                response.score[config_id] = float(weighted_score)
            return response

        except Exception as error:
            logging.error(f"failed to get weighted score.\n\t{error}")
            return CostComputation.create_compute_response(
                exit_code=1, message="failed to calculate weighted score.",
                iteration_id=iteration_id
            )

    def Close(self, request, context):
        if not request.token:
            response = cost_service_pb2.CloseResponse()
            response.status.code = 1
            response.status.message = "Service token not specified."
            return response

        SimClient.set_channel(flags.FLAGS.sim_service_url)
        status = SimClient.close(request.token)
        return status


def __main__(argv):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        compression=grpc.Compression.Gzip)
    cost_service_pb2_grpc.add_CostComputationServicer_to_server(
        CostComputation(), server
    )
    server.add_insecure_port(SERVER_PORT)
    server.start()

    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    app.run(__main__)
