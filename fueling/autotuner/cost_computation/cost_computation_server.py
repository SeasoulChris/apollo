#!/usr/bin/env python

# standard packages
from concurrent import futures
import os
import uuid

# third party packages
from absl import app
from absl import flags
import grpc

# Apollo-fuel packages
import modules.data.fuel.fueling.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import modules.data.fuel.fueling.autotuner.proto.cost_computation_service_pb2_grpc as cost_service_pb2_grpc

import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils
import fueling.common.logging as logging

# Flags
flags.DEFINE_enum(
    "running_mode", "TEST", ["TEST", "PROD"], "server running mode: TEST, PROD ."
)

SERVER_PORT = "[::]:50052"
TMP_ROOT_DIR = "/tmp/autotuner"


class CostComputation(cost_service_pb2_grpc.CostComputationServicer):
    """The Python implementation of the GRPC cost computation server."""

    def __init__(self):
        logging.info(f"Running server in {flags.FLAGS.running_mode} mode.")
        if flags.FLAGS.running_mode == "PROD":
            self.submit_job_cmd = "python ./tools/submit-job-to-k8s.py"
        else:
            self.submit_job_cmd = "./tools/submit-job-to-local.sh"

    def CreateResponse(self, exit_code, message="", score=None):
        response = cost_service_pb2.Response()
        response.status.code = exit_code
        response.status.message = message
        if score is not None:
            response.score = score
        return response

    def ComputeMracCost(self, request, context):
        if not request.git_info.commit_id:
            return self.CreateResponse(exit_code=1, message="Commit ID not specified.")

        training_id = uuid.uuid1().hex
        tmp_dir = f"{TMP_ROOT_DIR}/{training_id}"
        file_utils.makedirs(tmp_dir)

        # Save config to a local file
        proto_utils.write_pb_to_text_file(request, f"{tmp_dir}/request.pb.txt")

        # submit job
        cmd = (
            f"{self.submit_job_cmd} fueling/autotuner/cost_computation/mrac_cost_computation.py"
            f" --training_id={training_id} --commit_id={request.git_info.commit_id}"
        )
        # TODO: exit_code does not work so far, check abseil's app to see how to set exit code
        exit_code = os.system(cmd)
        if exit_code != 0:
            return self.CreateResponse(
                exit_code=exit_code, message="Error running mrac_cost_computation."
            )

        # read and return score
        try:
            with open(f"{tmp_dir}/score.out") as score_file:
                score = float(score_file.readline())
                return self.CreateResponse(exit_code=0, message="Done.", score=score)
        except Exception as error:
            logging.error(f"Failed to get weighted score.\n\t{error}")
            return self.CreateResponse(
                exit_code=1, message="Failed to get weighted score."
            )


def __main__(argv):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cost_service_pb2_grpc.add_CostComputationServicer_to_server(
        CostComputation(), server
    )
    server.add_insecure_port(SERVER_PORT)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    app.run(__main__)
