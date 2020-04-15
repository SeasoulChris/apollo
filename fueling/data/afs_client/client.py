#!/usr/bin/env python

from absl import flags
import os
import time

import grpc

from cyber_py3.record import RecordWriter
from fueling.common.base_pipeline import BasePipeline
from fueling.common.record.kinglong.cybertron.python.convert import transfer_localization_estimate
import apps.afs_data_service.proto.afs_data_service_pb2 as afs_data_service_pb2
import apps.afs_data_service.proto.afs_data_service_pb2_grpc as afs_data_service_pb2_grpc
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record.kinglong.proto.modules.localization_pose_pb2 as cybertron_localization_pose_pb2


class AfsClient(object):
    """Afs client."""

    def __init__(self, scan_table_name=None, message_table_name=None):
        """init common variables"""
        self.SERVER_URL = '180.76.53.252:50053'
        self.scan_table_name = scan_table_name or 'kinglong/auto_car/cyberecord'
        self.message_table_name = message_table_name or 'kinglong/auto_car'

    def convert_message(self, topic, message):
        """Check message format"""
        # TODO(all): This is for demonstration only.  Replace this with real conversion later.
        if topic == '/localization/100hz/localization_pose':
            loc = cybertron_localization_pose_pb2.LocalizationEstimate()
            loc.ParseFromString(message)
            apollo_loc = transfer_localization_estimate(loc)
            logging.info(F'localization coordinates after transfer: {apollo_loc.pose.position.x}')

    def scan_tasks(self, start_time, end_time):
        """Scan tasks by date range"""
        res = []
        columns='task_id,start_time,end_time'
        with grpc.insecure_channel(self.SERVER_URL) as channel:
            # Get scan result, it could be a list of record files
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.ScanRequest(
                table_name=self.scan_table_name,
                columns=columns,
                where='date >= {} and date <= {}'.format(start_time, end_time))
            response = stub.Scan(request)
            for resItem in response.records:
                task_id = resItem.task_id
                start_time = resItem.start_time
                end_time = resItem.end_time
                res.append((task_id, start_time, end_time))
                logging.info(F'task id: {task_id}, start_time: {start_time}, end_time: {end_time}')
        return res

    def transfer_messages(self, task_id, start_time, end_time, target_dir, topics='*'):
        with grpc.insecure_channel(self.SERVER_URL) as channel:
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            # Get ReadMessages result, it could be a stream of record messages
            request = afs_data_service_pb2.ReadMessagesRequest(
                task_id=task_id,
                start_time_second=start_time,
                end_time_second=end_time,
                table_name=self.message_table_name,
                topics=topics)
            response = stub.ReadMessages(request)
            # Write message to BOS
            target_dir = os.path.join(target_dir, task_id)
            file_utils.makedirs(target_dir)
            target_file = os.path.join(target_dir, F'{start_time}.record')
            writer = RecordWriter(0, 0)
            writer.open(target_file)
            for msg in response:
                logging.info(F'message topic: {msg.topic}')
                self.convert_message(msg.topic, msg.message)
                writer.write_message(msg.topic, msg.message, msg.timestamp)
            writer.close()


class AfsClientPipeline(BasePipeline):
    """AFS data transfer pipeline""" 
    def execute(self, task_id):
        """Connect to gRPC server, issue requests and get responses"""
        # TODO(all): for demonstation only for now, replace the params with real ones later
        target_dir = self.our_storage().abs_path('modules/data/planning')
        afs_client = AfsClient()
        afs_client.transfer_messages('KL056_20200326111402', 1585193067, 1585193068, target_dir)
        logging.info('done executing')
        time.sleep(60*2)
        return 1

    def run(self):
        """Run."""
        # Multiple tasks to get messages from gRPC service in parallel
        tasks_count = 1
        total_messages_num = self.to_rdd(range(tasks_count)).map(self.execute).sum()
        logging.info(F'total messages number: {total_messages_num}')


if __name__ == '__main__':
    AfsClientPipeline().main()

