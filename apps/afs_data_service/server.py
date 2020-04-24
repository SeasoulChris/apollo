#!/usr/bin/env python

from concurrent import futures
import json
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
        print('Running AfsDataTransfer server.')
        self.adb_client = AdbClient()
        # TODO(weixiao): Replace IP with a list
        self.adb_client.set_config('adb.server.hosts', '10.197.199.17:8010')
        self.adb_client.set_config('adb.export_server.hosts', '10.90.222.37:8000')
        self.adb_client.set_config('adb.user', os.environ.get('ADB_SDK_USER'))
        self.adb_client.set_config('adb.pass', os.environ.get('ADB_SDK_PASSWD'))
        print('AfsDataTransfer server running with adbsdk client setup.')

    def Scan(self, request, context):
        """Scan"""
        print('scanning table {} with where {}'.format(request.table_name, request.where))
        scan = cmm.Scan(table_name=request.table_name,
                        where=request.where,
                        columns=request.columns if request.columns else '*',
                        order=request.order,
                        limit=request.limit)
        scan_result_iterator = self.adb_client.scan(scan)
        response = afs_data_service_pb2.ScanResponse()
        for scan_result in scan_result_iterator:
            if not scan_result.success:
                print('exception occurred: {}'.format(scan_result.errMessage))
                continue
            json_rets = {}
            for k, v in scan_result.meta.items():
                json_rets[k] = self.get_value(v)
            response.records.append(json.dumps(json_rets))
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
        for topic, message, data_type, timestamp in messages:
            if request.skip_topics != '' and any(topic.find(x) != -1 for x in skip_topics):
                print('skipping topic: {}'.format(topic))
                continue
            response = afs_data_service_pb2.ReadMessagesResponse()
            response.topic = topic
            response.message = message if request.with_data else b''
            response.message_size = len(message)
            response.data_type = data_type
            response.timestamp = timestamp
            yield response

    def get_value(self, data):
        """get scan result meta column value"""
        if data.HasField('int'):
            return data.int
        if data.HasField('str'):
            return data.str
        if data.HasField('long'):
            return data.long
        if data.HasField('double'):
            return data.double


def __main__(argv):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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

