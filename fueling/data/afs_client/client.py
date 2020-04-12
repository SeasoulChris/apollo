#!/usr/bin/env python

import time

from absl import flags
import grpc

from fueling.common.base_pipeline import BasePipeline
import apps.afs_data_service.proto.afs_data_service_pb2 as afs_data_service_pb2
import apps.afs_data_service.proto.afs_data_service_pb2_grpc as afs_data_service_pb2_grpc
import fueling.common.logging as logging


class AfsClient(BasePipeline):
    """Afs client."""
    def execute(self, task_id):
        """Connect to gRPC server, issue requests and get responses"""
        SERVER_URL = '180.76.53.252:50053'
        messages_num = 0
        with grpc.insecure_channel(SERVER_URL) as channel:
            # Get scan result, it could be a list of record files
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.ScanRequest(dates=['123', '456'])
            response = stub.Scan(request)
            logging.info(F'task id: {task_id}, scan response: {response}')

            # Get ReadMessages result, it could be a stream of record messages
            request = afs_data_service_pb2.ReadMessagesRequest(record_files=['record1', 'record2'])
            response = stub.ReadMessages(request)
            for message in response:
                logging.info(F'task id: {task_id}, msg: {message.status}')
                messages_num += 1
        # Wait a while for checking executor logs, and return message number for this executor
        logging.info(F'executor {task_id}: got {messages_num} messages')
        time.sleep(10)
        return messages_num

        
    def run(self):
        """Run."""
        # Multiple tasks to get messages from gRPC service in parallel
        tasks_count = 100
        total_messages_num = self.to_rdd(range(tasks_count)).map(self.execute).sum()
        logging.info(F'total messages number: {total_messages_num}')


if __name__ == '__main__':
    AfsClient().main()
