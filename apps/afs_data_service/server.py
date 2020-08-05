#!/usr/bin/env python2
"""
TODO(Data): If we got an Python3 ADB SDK, we can leverage Bazel to build the executable package.
"""

from concurrent import futures
import json
import logging
import os
import time

from absl import app
import grpc

from adbsdk import cmm
from adbsdk.adb_client import AdbClient
from adbsdk.dump.record_dump import RecordDump
import afs_data_service_pb2
import afs_data_service_pb2_grpc


class AfsDataTransfer(afs_data_service_pb2_grpc.AfsDataTransferServicer):
    """The Python implementation of the GRPC data transfer server."""

    def __init__(self):
        """Init"""
        logging.info('Running AfsDataTransfer server.')
        self.adb_client = AdbClient()
        # TODO(weixiao): Replace IP with a list
        self.adb_client.set_config('adb.server.hosts', '10.197.198.23:8010')
        self.adb_client.set_config('adb.export_server.hosts', '10.104.102.30:8000')
        self.adb_client.set_config('adb.user', os.environ.get('ADB_SDK_USER'))
        self.adb_client.set_config('adb.pass', os.environ.get('ADB_SDK_PASSWD'))
        logging.info('AfsDataTransfer server running with adbsdk client setup.')

    def Scan(self, request, context):
        """Scan"""
        logging.info('scanning table {} with where {}'.format(request.table_name, request.where))
        scan = cmm.Scan(table_name=request.table_name,
                        where=request.where,
                        columns=request.columns if request.columns else '*',
                        order=request.order,
                        limit=request.limit)
        scan_result_iterator = self.adb_client.scan(scan)
        response = afs_data_service_pb2.ScanResponse()
        for scan_result in scan_result_iterator:
            if not scan_result.success:
                logging.error('exception occurred: {}'.format(scan_result.errMessage))
                continue
            json_rets = {}
            for k, v in scan_result.meta.items():
                json_rets[k] = self._get_value(v)
            response.records.append(json.dumps(json_rets))
        logging.info('scanned table {} with where {}'.format(request.table_name, request.where))
        return response

    def ReadMessages(self, request, context):
        """Dump and ReadMessages"""
        record_dump = RecordDump(self.adb_client)
        messages = record_dump.read_messages(
            task_id=request.task_id,
            topics=request.topics if request.topics else '*',
            start_time_s=request.start_time_second,
            end_time_s=request.end_time_second,
            namespace=request.namespace)
        skip_topics = request.skip_topics.split(',')
        logging.info('read messages for {} with topics {}, skipping {}'.format(
            request.task_id, request.topics, skip_topics))
        for topic, message, data_type, timestamp in messages:
            if request.skip_topics != '' and any(topic.find(x) != -1 for x in skip_topics):
                continue
            response = afs_data_service_pb2.ReadMessagesResponse()
            response.topic = topic
            response.message = message if request.with_data else b''
            response.message_size = len(message)
            response.data_type = data_type
            response.timestamp = timestamp
            yield response
        logging.info('finished read messages for {}'.format(request.task_id))

    def GetLogs(self, request, context):
        """Retrieve logs from particular task"""
        log_path = '{}/{}/{}/{}/otherlog/cmptnode/xlog/log'.format(
            request.log_table_name,
            request.vehicle_id,
            request.log_date,
            request.task_id)
        log_names = request.log_names.split(',')
        logging.info('getting {} from path: {}'.format(log_names, log_path))
        response = afs_data_service_pb2.GetLogsResponse()
        log_files = self.adb_client.path_ls(log_path)
        if log_files.success:
            for log_file_path in log_files.paths:
                if (log_file_path.type != 'd'
                        and any(log_file_path.path.find(x) != -1 for x in log_names)):
                    response.log_file_name = log_file_path.path
                    response.log_content = self._retrieve_file_content(log_file_path.path)
                    logging.info('got log for: {}'.format(log_file_path.path))
                    yield response
        logging.info('finished getting {} from path: {}'.format(log_names, log_path))

    def _get_value(self, data):
        """get scan result meta column value"""
        if data.HasField('int'):
            return data.int
        if data.HasField('str'):
            return data.str
        if data.HasField('long'):
            return data.long
        if data.HasField('double'):
            return data.double

    def _retrieve_file_content(self, file_path):
        """Download file, retrieve its content and then delete"""
        # Download first to tmp
        DST_FOLDER = '/tmp'
        self.adb_client.path_get(file_path, DST_FOLDER)
        # Read its content into memory
        local_file_name = os.path.join(DST_FOLDER, os.path.basename(file_path))
        file_content = None
        with open(local_file_name, 'r') as local_file:
            file_content = local_file.read()
        # Remove the file to release space
        os.remove(local_file_name)
        return file_content


def __main__(argv):
    MAX_WORKERS = 20
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    afs_data_service_pb2_grpc.add_AfsDataTransferServicer_to_server(
        AfsDataTransfer(), server)
    server.add_insecure_port('[::]:50053')
    server.start()

    ONE_DAY_IN_SECONDS = 60 * 60 * 24

    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    app.run(__main__)
