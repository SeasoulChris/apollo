#!/usr/bin/env python

# standard packages
from concurrent import futures
import json
import os
import time
import uuid

# third party packages
from absl import app
from absl import flags
import grpc

from apps.k8s.spark_submitter.client import SparkSubmitterClient
import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

ONE_DAY_IN_SECONDS = 60 * 60 * 24

# Flags
flags.DEFINE_enum(
    "running_mode", "TEST", ["TEST", "PROD"],
    "server running mode: TEST, PROD."
    "PROD mode submits spark job to specified spark_service_url"
)
flags.DEFINE_string(
    "sim_service_url", "localhost:50051", "channel url to sim service"
)

SERVER_PORT = "[::]:50052"

TMP_ROOT_DIR = "/mnt/bos/autotuner"


class CostComputation(cost_service_pb2_grpc.CostComputationServicer):
    """The Python implementation of the GRPC cost computation server."""

    def __init__(self):
        logging.info(f"Running server in {flags.FLAGS.running_mode} mode.")

        if flags.FLAGS.running_mode != "PROD":
            self.submit_job = self.SubmitJobAtLocal
        else:
            self.submit_job = self.SubmitJobToK8s

    def CreateResponse(self, exit_code, message="", score=None):
        response = cost_service_pb2.Response()
        response.status.code = exit_code
        response.status.message = message
        if score is not None:
            response.score = score
        return response

    def SubmitJobAtLocal(self, options):
        job_cmd = "bazel run //fueling/learning/autotuner/cost_computation:mrac_cost_computation"
        option_strings = [F"--{name}={value}" for (name, value) in options.items()]
        cmd = f"cd /fuel; {job_cmd} -- {' '.join(option_strings)}"

        # TODO: exit_code does not work so far, check abseil's app to see how to set exit code
        exit_code = os.system(cmd)
        return os.WEXITSTATUS(exit_code) == 0

    def SubmitJobToK8s(self, options):
        entrypoint = "fueling/learning/autotuner/cost_computation/mrac_cost_computation.py"
        options['mnt_root_dir'] = "/mnt/bos-rw"
        client = SparkSubmitterClient(entrypoint, {}, options)
        client.submit()
        return True

    def ComputeMracCost(self, request, context):
        if not request.git_info.commit_id:
            return self.CreateResponse(exit_code=1, message="Commit ID not specified.")

        training_id = uuid.uuid4().hex
        tmp_dir = f"{TMP_ROOT_DIR}/{training_id}"
        file_utils.makedirs(tmp_dir)

        # Save config to a local file
        proto_utils.write_pb_to_text_file(request, f"{tmp_dir}/request.pb.txt")

        # submit job
        options = {
            "running_mode": flags.FLAGS.running_mode,
            "sim_service_url": flags.FLAGS.sim_service_url,
            "commit_id": request.git_info.commit_id,
            "training_id": training_id,
        }

        if not self.submit_job(options):
            return self.CreateResponse(
                exit_code=1, message="failed to run mrac_cost_computation."
            )

        # read and return score
        try:
            with open(f"{tmp_dir}/scores.out") as score_file:
                scores = json.loads(score_file.readline())

            response = self.CreateResponse(exit_code=0, message="Done.")
            for (config_id, weighted_score) in scores.items():
                response.score[config_id] = float(weighted_score)
            return response

        except Exception as error:
            logging.error(f"failed to get weighted score.\n\t{error}")
            return self.CreateResponse(
                exit_code=1, message="failed to calculate weighted score."
            )


def __main__(argv):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
