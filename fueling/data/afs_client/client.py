#!/usr/bin/env python

import os

import grpc

from cyber_py3.record import RecordWriter
from fueling.common.record.kinglong.cybertron.python.convert import transfer_localization_estimate
import apps.afs_data_service.proto.afs_data_service_pb2 as afs_data_service_pb2
import apps.afs_data_service.proto.afs_data_service_pb2_grpc as afs_data_service_pb2_grpc
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record.kinglong.proto.modules.localization_pose_pb2 as cybertron_localization_pose_pb2


class AfsClient(object):
    """Afs client."""

    def __init__(self, scan_table_name, message_namespace):
        """init common variables"""
        self.SERVER_URL = '180.76.53.252:50053'
        self.GRPC_OPTIONS = [
            ('grpc.max_send_message_length', 512 * 1024 * 1024),
            ('grpc.max_receive_message_length', 512 * 1024 * 1024)
        ]
        self.scan_table_name = scan_table_name
        self.message_namespace = message_namespace
        self.respect_existing = True

    def convert_message(self, topic, message):
        """Check message format"""
        # TODO(all): This is for demonstration only.  Replace this with real conversion later.
        if topic == '/localization/100hz/localization_pose':
            loc = cybertron_localization_pose_pb2.LocalizationEstimate()
            loc.ParseFromString(message)
            apollo_loc = transfer_localization_estimate(loc)
            logging.info(F'localization coordinates after transfer: {apollo_loc.pose.position.x}')

    def scan_tasks(self, query_date):
        """Scan tasks by date"""
        res = []
        columns='task_id,start_time,end_time'
        with grpc.insecure_channel(self.SERVER_URL, self.GRPC_OPTIONS) as channel:
            # Get scan result, it could be a list of record files
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.ScanRequest(
                table_name=self.scan_table_name,
                columns=columns,
                where='date = {}'.format(query_date))
            response = stub.Scan(request)
            for resItem in response.records:
                task_id = resItem.task_id
                start_time = resItem.start_time
                end_time = resItem.end_time
                res.append((task_id, start_time, end_time))
                logging.info(F'task id: {task_id}, start_time: {start_time}, end_time: {end_time}')
        return res

    def transfer_messages(self, task_params, target_dir, skip_topics, topics='*'):
        """Read and transfer afs messages into apollo format, then insert them into bos"""
        task_id, start_time, end_time = task_params
        target_dir = os.path.join(target_dir, task_id)
        file_utils.makedirs(target_dir)
        target_file = os.path.join(target_dir, F'{start_time}.record')
        logging.info(F'writing to target file: {target_file}')
        if os.path.exists(target_file) and self.respect_existing:
            logging.info(F'target file {target_file} exists already, skip it')
            return target_file
        with grpc.insecure_channel(self.SERVER_URL, self.GRPC_OPTIONS) as channel:
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            # Get ReadMessages result, it could be a stream of record messages
            request = afs_data_service_pb2.ReadMessagesRequest(
                task_id=task_id,
                start_time_second=start_time,
                end_time_second=end_time,
                namespace=self.message_namespace,
                topics=topics,
                skip_topics=skip_topics,
                with_data=True)
            response = stub.ReadMessages(request)
            # Write message to BOS
            writer = RecordWriter(0, 0)
            writer.open(target_file)
            for msg in response:
                self.convert_message(msg.topic, msg.message)
                writer.write_message(msg.topic, msg.message, msg.timestamp)
            writer.close()
        return target_file

    def get_topics(self, task_id, start_time, end_time):
        """Get topics of task"""
        topics = []
        with grpc.insecure_channel(self.SERVER_URL, self.GRPC_OPTIONS) as channel:
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.ReadMessagesRequest(
                task_id=task_id,
                start_time_second=start_time,
                end_time_second=end_time,
                namespace=self.message_namespace,
                with_data=False)
            response = stub.ReadMessages(request)
            for msg in response:
                topics.append((msg.topic, msg.message_size))
                logging.info((msg.topic, msg.message_size))
        return topics


