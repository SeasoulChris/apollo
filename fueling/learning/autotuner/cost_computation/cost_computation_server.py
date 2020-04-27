#!/usr/bin/env python

# standard packages
from concurrent import futures
from datetime import datetime
import json
import os
import threading
import time
import uuid

# third party packages
from absl import app
from absl import flags
import grpc

from fueling.learning.autotuner.client.sim_client import SimClient
from fueling.learning.autotuner.cost_computation.job.local_cost_job import LocalCostJob
from fueling.learning.autotuner.cost_computation.job.k8s_cost_job import K8sCostJob
import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

ONE_DAY_IN_SECONDS = 60 * 60 * 24

MAX_SPARK_WORKERS = 10

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
            self.CostJob = K8sCostJob
        else:
            self.CostJob = LocalCostJob

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

    def get_service_dir(self, token):
        return f"{TMP_ROOT_DIR}/{token}"

    def Initialize(self, request, context):
        if not request.git_info.commit_id:
            return CostComputation.create_init_response(1, "Commit ID not specified.")
        if not request.scenario_id:
            return CostComputation.create_init_response(1, "Scenario(s) not specified.")
        tic_start = time.perf_counter()

        # set callback
        init_sim_event = threading.Event()
        stop_event = threading.Event()

        def on_rpc_done():
            stop_event.set()
            if init_sim_event.is_set():
                SimClient.set_channel(flags.FLAGS.sim_service_url)
                SimClient.close(service_token)
        context.add_callback(on_rpc_done)

        # Save config to a local file
        service_token = f"tuner-{uuid.uuid4().hex}"
        tmp_dir = self.get_service_dir(service_token)
        file_utils.makedirs(tmp_dir)
        proto_utils.write_pb_to_text_file(request, f"{tmp_dir}/init_request.pb.txt")

        # check if cancelled
        if stop_event.is_set():
            return CostComputation.create_init_response(
                1, "request cancelled", service_token,
            )

        # init sim
        num_workers = len(request.scenario_id)
        SimClient.set_channel(flags.FLAGS.sim_service_url)
        init_sim_event.set()
        status = SimClient.initialize(
            service_token,
            request.git_info,
            num_workers,
            request.dynamic_model)
        init_sim_event.clear()

        logging.info(f"Timer: total init - {time.perf_counter() - tic_start:0.04f} sec")
        return CostComputation.create_init_response(
            status.code, status.message, service_token,
        )

    def ComputeCost(self, request, context):
        if not request.token:
            return CostComputation.create_compute_response(
                exit_code=1, message="Service token not specified.")
        tic_start = time.perf_counter()

        # set callback
        stop_event = threading.Event()
        compute_event = threading.Event()
        job = None

        def on_rpc_done():
            stop_event.set()
            if job and compute_event.is_set():
                job.cancel()
        context.add_callback(on_rpc_done)

        def get_num_spark_workers(service_dir):
            request_pb2 = cost_service_pb2.InitRequest()
            proto_utils.get_pb_from_text_file(
                os.path.join(service_dir, "init_request.pb.txt"), request_pb2,
            )
            return min(len(request_pb2.scenario_id), MAX_SPARK_WORKERS)

        # Save config to a local file
        iteration_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        service_dir = self.get_service_dir(request.token)
        iteration_dir = os.path.join(service_dir, iteration_id)
        file_utils.makedirs(iteration_dir)
        proto_utils.write_pb_to_text_file(
            request, os.path.join(
                iteration_dir, "compute_request.pb.txt"))
        if stop_event.is_set():
            return CostComputation.create_compute_response(
                exit_code=1, message="Request cancelled")

        # submit job
        options = {
            "sim_service_url": flags.FLAGS.sim_service_url,
            "token": request.token,
            "iteration_id": iteration_id,
            "workers": get_num_spark_workers(service_dir),
        }
        if request.cost_computation_conf_filename:
            options['cost_computation_conf_filename'] = request.cost_computation_conf_filename

        compute_event.set()
        try:
            job = self.CostJob()
            job.submit(options)
        except Exception as error:
            logging.error(f'Job failed: {error}')
            return CostComputation.create_compute_response(
                exit_code=1, message="Failed to submit job."
            )
        compute_event.clear()
        if stop_event.is_set():
            return CostComputation.create_compute_response(
                exit_code=1, message="Request cancelled")

        # read and return score
        try:
            with open(os.path.join(iteration_dir, "scores.out")) as score_file:
                scores = json.loads(score_file.readline())

            response = CostComputation.create_compute_response(
                exit_code=0, message="Done.", iteration_id=iteration_id)
            for (config_id, weighted_score) in scores.items():
                response.score[config_id] = float(weighted_score)

            logging.info(f"Timer: total compute - {time.perf_counter() - tic_start:0.04f} sec")
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
