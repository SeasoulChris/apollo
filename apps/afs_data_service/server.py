#!/usr/bin/env python
from concurrent import futures
import os
import time

from absl import app
import grpc

from adbsdk import cmm
from adbsdk.adb_client import AdbClient
from adbsdk.dump.record_dump import RecordDump
import afs_data_service_pb2 as afs_data_service_pb2
import afs_data_service_pb2_grpc as afs_data_service_pb2_grpc


class AfsDataTransfer(afs_data_service_pb2_grpc.AfsDataTransferServicer):
    """The Python implementation of the GRPC data transfer server."""

    def __init__(self):
        """Init"""
        print('Running AfsDataTransfer server.')
        self.adb_client = AdbClient()
        self.adb_client.set_config('adb.server.bns', 'adb-server-online.IDG.yq')
        self.adb_client.set_config('adb.user', os.environ.get('ADB_SDK_USER'))
        self.adb_client.set_config('adb.pass', os.environ.get('ADB_SDK_PASSWD'))
        print('AfsDataTransfer server running with adbsdk client setup.')

    def Scan(self, request, context):
        """Scan"""
        scan = cmm.Scan(table_name='kinglong/auto_car/cyberecord',
                        where='date = 20200330')
        scan_result_iterator = self.adb_client.scan(scan)
        for scan_result in scan_result_iterator:
            if scan_result.success is not True:
                #raise Exception('exception occurred: %s' % (scan_result.errMessage, ))
                print('exception occurred: {}'.format(scan_result.errMessage))
            print('rowKey:{}, task_id: {}, stime: {}, etime: {}'.format(
                  scan_result.rowKey,
                  scan_result.meta['task_id'],
                  scan_result.meta['start_time'],
                  scan_result.meta['end_time']))
        response = afs_data_service_pb2.ScanResponse()
        return response

    def ReadMessages(self, request, context):
        """Dump and ReadMessages"""
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
    
    ONE_DAY_IN_SECONDS = 60 * 60 * 24

    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    app.run(__main__)
