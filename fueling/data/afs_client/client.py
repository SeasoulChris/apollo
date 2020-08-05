#!/usr/bin/env python

import io
import json
import os
import sys

import grpc

from fueling.data.afs_client.afs_record_writer import AfsRecordWriter
import apps.afs_data_service.proto.afs_data_service_pb2 as afs_data_service_pb2
import apps.afs_data_service.proto.afs_data_service_pb2_grpc as afs_data_service_pb2_grpc
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


# print chinese characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class AfsClient(object):

    """Afs client."""

    def __init__(self):
        """init common variables"""
        self.SERVER_URL = '180.76.117.150:50053'
        self.GRPC_OPTIONS = [
            ('grpc.max_send_message_length', 512 * 1024 * 1024),
            ('grpc.max_receive_message_length', 512 * 1024 * 1024)
        ]
        self.respect_existing = True
        # Timeout in seconds for grpc calls
        self.timeout = 10 * 60

    def scan(self, tablename, columns, where):
        """Scan data from table"""
        logging.info(F'tablename:{tablename}, columns:{columns}, where:{where}')
        res = []
        with grpc.insecure_channel(self.SERVER_URL, self.GRPC_OPTIONS) as channel:
            # Get scan result, it could be a list of record files
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.ScanRequest(
                table_name=tablename,
                columns=columns,
                where=where)
            response = stub.Scan(request, timeout=self.timeout)
            for resItem in response.records:
                json_rets = json.loads(resItem)
                if json_rets is None:
                    continue
                res.append(json_rets)
        return res

    def scan_tasks(self, table_name, query_date):
        """Scan tasks by date"""
        res = []
        columns = 'task_id,start_time,end_time'
        where = 'date = {}'.format(query_date)
        query_ret = self.scan(table_name, columns, where)
        for item in query_ret:
            task_id = item.get('task_id', '')
            start_time = int(float(str(item.get('start_time', ''))[:10]))
            end_time = int(float(str(item.get('end_time', ''))[:10]))
            res.append((task_id, start_time, end_time))
            logging.info(F'task id: {task_id}, start_time: {start_time}, end_time: {end_time}')
        return res

    def scan_keydata(self, table_name, task_id):
        """Scan key data of specific task_id"""
        res = []
        table_name = table_name
        columns = 'capture_place,region_id,task_purpose'
        where = 'task_id = {}'.format(task_id)
        query_ret = self.scan(table_name, columns, where)
        for item in query_ret:
            capture_place = item.get('capture_place', '')
            region_id = item.get('region_id', '')
            task_purpose = item.get('task_purpose', '')
            res.append((task_id, capture_place, region_id, task_purpose))
            logging.info(F'task id: {task_id}, '
                         F'capture_place: {capture_place}, '
                         F'region_id: {region_id}, '
                         F'task_purpose: {task_purpose}')
        return res

    def scan_region_info(self):
        """Scan name of region_id"""
        res = []
        table_name = self.region_info_table_name
        columns = 'region_id,name'
        where = ''
        query_ret = self.scan(table_name, columns, where)
        for item in query_ret:
            region_id = item.get('region_id', '')
            region_name = item.get('name', '')
            res.append((region_id, region_name))
        return res

    def scan_map_area(self):
        """Scan area name by map_area_id"""
        res = []
        table_name = self.map_area_table_name
        columns = 'map_area_id,map_area_name'
        where = ''
        query_ret = self.scan(table_name, columns, where)
        for item in query_ret:
            map_area_id = item.get('map_area_id', '')
            map_area_name = item.get('map_area_name', '')
            res.append((map_area_id, map_area_name))
            logging.info(F'map_area_id: {map_area_id}, map_area_name: {map_area_name}')
        return res

    def transfer_messages(self, task_params, message_namespace, skip_topics, topics='*'):
        """Read and transfer afs messages into apollo format, then insert them into bos"""
        task_id, ((start_time, end_time), target_dir) = task_params
        target_file = os.path.join(target_dir, F'{start_time}.record')
        logging.info(F'writing to target file: {target_file}, message table name: {message_namespace}')
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
                namespace=message_namespace,
                topics=topics,
                skip_topics=skip_topics,
                with_data=True)
            response = stub.ReadMessages(request, timeout=self.timeout)
            # Write message to BOS
            writer = AfsRecordWriter(target_file)
            for cyber_message in response:
                writer.write_message(cyber_message)
            writer.close()
        return target_file

    def get_topics(self, message_namespace, task_id, start_time, end_time):
        """Get topics of task"""
        topics = []
        with grpc.insecure_channel(self.SERVER_URL, self.GRPC_OPTIONS) as channel:
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.ReadMessagesRequest(
                task_id=task_id,
                start_time_second=start_time,
                end_time_second=end_time,
                namespace=message_namespace,
                with_data=False)
            response = stub.ReadMessages(request)
            for msg in response:
                topics.append((msg.topic, msg.message_size))
                logging.info((msg.topic, msg.message_size))
        return topics

    def get_logs(self, log_table, task_target, log_names):
        """Get logs of tasks"""
        task_id, log_dir = task_target
        task_parts = task_id.split('_')
        vehicle_id, log_date = task_parts[0], task_parts[1][:8]
        logging.info(F'task_id: {task_id}, target log dir: {log_dir}')
        with grpc.insecure_channel(self.SERVER_URL, self.GRPC_OPTIONS) as channel:
            stub = afs_data_service_pb2_grpc.AfsDataTransferStub(channel)
            request = afs_data_service_pb2.GetLogsRequest(
                task_id=task_id,
                log_table_name=log_table,
                vehicle_id=vehicle_id,
                log_date=log_date,
                log_names=log_names)
            response = stub.GetLogs(request, timeout=self.timeout)
            for log in response:
                log_file_path = os.path.join(log_dir, os.path.basename(log.log_file_name))
                logging.info(F'writing log file: {log_file_path}')
                file_utils.makedirs(log_dir)
                with open(log_file_path, 'w') as log_file:
                    log_file.write(log.log_content.encode('utf-8').decode())
