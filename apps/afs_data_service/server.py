#!/usr/bin/env python
from concurrent import futures
import time

from absl import app
import grpc

import afs_data_service_pb2 as afs_data_service_pb2
import afs_data_service_pb2_grpc as afs_data_service_pb2_grpc

ONE_DAY_IN_SECONDS = 60 * 60 * 24


class AfsDataTransfer(afs_data_service_pb2_grpc.AfsDataTransferServicer):
    """The Python implementation of the GRPC data transfer server."""

    def __init__(self):
        print('Running AfsDataTransfer server.')

    def Scan(self, request, context):
        response = afs_data_service_pb2.ScanResponse()
        return response

    def ReadMessages(self, request, context):
        for x in range(10):
            response = afs_data_service_pb2.ReadMessagesResponse(status=x)
            yield response

def __main__(argv):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    afs_data_service_pb2_grpc.add_AfsDataTransferServicer_to_server(
        AfsDataTransfer(), server
    )
    server.add_insecure_port('[::]:50053')
    server.start()

    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    app.run(__main__)
