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
            request = afs_data_service_pb2.ScanRequest(
                table_name='kinglong/auto_car/cyberecord',
                where='task_id = KL056_20200316141342')
            response = stub.Scan(request)
            for resItem in response.records:
                rowKey = resItem.rowKey
                task_id = resItem.task_id
                start_time = resItem.start_time
                end_time = resItem.end_time
                logging.info(
                    F'rowKey: {rowKey}, task id: {task_id}, start_time: {start_time}, end_time: {end_time}')

            # Get ReadMessages result, it could be a stream of record messages
            request = afs_data_service_pb2.ReadMessagesRequest(
                task_id='KL056_20200326111402',
                start_time_second=1585193067,
                end_time_second=1585193068,
                table_name='kinglong/auto_car')
            response = stub.ReadMessages(request)
            for message in response:
                topic = message.topic
                messagetext = message.message
                data_type = message.data_type
                timestamp = message.timestamp
                logging.info(
                    F'task id: {topic}, data_type: {data_type}, timestamp: {timestamp}')
                messages_num += 1
        # Wait a while for checking executor logs, and return message number
        # for this executor
        logging.info(F'got {messages_num} messages')
        time.sleep(10)
        return messages_num

    def run(self):
        """Run."""
        # Multiple tasks to get messages from gRPC service in parallel
        tasks_count = 1
        total_messages_num = self.to_rdd(
            range(tasks_count)).map(
            self.execute).sum()
        logging.info(F'total messages number: {total_messages_num}')


if __name__ == '__main__':
    AfsClient().main()
