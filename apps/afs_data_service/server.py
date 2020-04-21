#!/usr/bin/env python

from concurrent import futures
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
            print('rowKey:{}, task_id: {}, stime: {}, etime: {}'.format(
                  scan_result.rowKey,
                  scan_result.meta['task_id'].str,
                  scan_result.meta['start_time'].str,
                  scan_result.meta['end_time'].str))
            responseItem = afs_data_service_pb2.ScanResponseItem()
            responseItem.rowKey = scan_result.rowKey
            responseItem.task_id = scan_result.meta['task_id'].str
            responseItem.start_time = int(float((scan_result.meta['start_time'].str)))
            responseItem.end_time = int(float((scan_result.meta['end_time'].str)))
            response.records.append(responseItem)
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
            print('task_id:{}, topic:{}, message_size: {}, data_type:{}, timestamp:{}'.format(
                  request.task_id, topic, len(message), data_type, timestamp))
            if any(topic.find(x) != -1 for x in skip_topics):
                continue
            response = afs_data_service_pb2.ReadMessagesResponse()
            response.topic = topic
            response.message = message if request.with_data else b''
            response.message_size = len(message)
            response.data_type = data_type
            response.timestamp = timestamp
            yield response


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

