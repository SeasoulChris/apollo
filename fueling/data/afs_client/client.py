#!/usr/bin/env python

import time
from absl import flags
import grpc
import math

from fueling.common.base_pipeline import BasePipeline
import apps.afs_data_service.proto.afs_data_service_pb2 as afs_data_service_pb2
import apps.afs_data_service.proto.afs_data_service_pb2_grpc as afs_data_service_pb2_grpc
import fueling.common.logging as logging
import os
import fueling.common.record.kinglong.proto.modules.localization_pose_pb2 as localization_pose_pb2
import modules.localization.proto.localization_pb2 as apollo_localization_pb2
from cyber_py3.record import RecordWriter
import fueling.common.file_utils as file_utils

def transfer_localization_estimate(loc):
    """transfer localization estimate"""
    apollo_loc = apollo_localization_pb2.LocalizationEstimate()

    apollo_loc.header.timestamp_sec = loc.header.timestamp_sec
    print('loc.header.timestamp_sec:{}'.format(loc.header.timestamp_sec))
    apollo_loc.pose.position.x = loc.pose.position.x
    print('loc.pose.position.x:{}'.format(loc.pose.position.x))
    apollo_loc.pose.position.y = loc.pose.position.y
    print('loc.pose.position.y:{}'.format(loc.pose.position.y))
    apollo_loc.pose.position.z = loc.pose.position.z
    print('loc.pose.position.z:{}'.format(loc.pose.position.z))
    apollo_loc.pose.orientation.qw = loc.pose.orientation.qw
    apollo_loc.pose.orientation.qx = loc.pose.orientation.qx
    apollo_loc.pose.orientation.qy = loc.pose.orientation.qy
    apollo_loc.pose.orientation.qz = loc.pose.orientation.qz

    heading = math.atan2(2 * (loc.pose.orientation.qw * loc.pose.orientation.qz +
                              loc.pose.orientation.qx * loc.pose.orientation.qy),
                         1 - 2 * (loc.pose.orientation.qy ** 2 + loc.pose.orientation.qz **
                                  2)) \
        + math.pi / 2
    apollo_loc.pose.heading = heading - 2 * math.pi if heading > math.pi else heading

    apollo_loc.pose.linear_velocity.x = loc.pose.linear_velocity.x
    apollo_loc.pose.linear_velocity.y = loc.pose.linear_velocity.y
    apollo_loc.pose.linear_velocity.z = loc.pose.linear_velocity.z

    return apollo_loc


class AfsClient(BasePipeline):
    """Afs client."""

    def __init__(self):
        """init common variables"""
        self.SERVER_URL = '180.76.53.252:50053'

    def check_transfer(self, topic, message):
        """Check message format"""
        if topic == '/localization/100hz/localization_pose':
            loc = localization_pose_pb2.LocalizationEstimate()
            loc.ParseFromString(message.message)
            # check if it throws an exception
            apollo_loc = transfer_localization_estimate(loc)

    def scan_tasks(self, begin, end,
                   columns='task_id,start_time,end_time',
                   tablename='kinglong/auto_car/cyberecord'):
        """Scan tasks by date range"""
        res = []
        with grpc.insecure_channel(self.SERVER_URL) as channel:
            # Get scan result, it could be a list of record files
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.ScanRequest(
                table_name=tablename,
                columns=columns,
                where='date >= {} and date <= {}'.format(begin, end))
            response = stub.Scan(request)
            for resItem in response.records:
                task_id = resItem.task_id
                start_time = resItem.start_time
                end_time = resItem.end_time
                res.append((task_id, start_time, end_time))
                logging.info(F'task id: {task_id}, start_time: {start_time}, end_time: {end_time}')
        return res

    def save_message(
            self,
            task_id,
            start_time,
            end_time,
            topics='*',
            table_name='kinglong/auto_car'):
        with grpc.insecure_channel(self.SERVER_URL) as channel:
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            # Get ReadMessages result, it could be a stream of record messages
            request = afs_data_service_pb2.ReadMessagesRequest(
                task_id=task_id,
                start_time_second=start_time,
                end_time_second=end_time,
                table_name=table_name,
                topics=topics)
            response = stub.ReadMessages(request)
            # Write message to BOS
            target_file = os.path.join(
                self.our_storage().abs_path('apollo-platform-fuel/modules/data/planning'),
                '{}.{}.record'.format(task_id, start_time))
            writer = RecordWriter(0, 0)
            try:
                writer.open(target_file)
                for msg in response:
                    writer.write_message(msg.topic, msg.message, msg.timestamp)
                    # uncomment below to check message format
                    # self.check_transfer(topic, msg.message)
            except Exception as e:
                logging.error('Failed to write to target file {}: {}'.format(target_file, e))
                return None
            finally:
                writer.close()

    def execute(self, task_id):
        """Connect to gRPC server, issue requests and get responses"""
        self.save_message('KL056_20200326111402', 1585193067, 1585193068)
        # Wait a while for checking executor logs, and return message number
        # for this executor
        logging.info(F'got messages')
        time.sleep(10)
        return 1

    def run(self):
        """Run."""
        # Multiple tasks to get messages from gRPC service in parallel
        tasks_count = [1]
        total_messages_num = self.to_rdd(
            tasks_count).map(
            self.execute).sum()
        logging.info(F'total messages number: {total_messages_num}')


if __name__ == '__main__':
    AfsClient().main()

