#!/usr/bin/env python3

###############################################################################
# Copyright 2020 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import argparse
import atexit
import logging
import os
import sys
import time

#from common.logger import Logger
from cyber.python.cyber_py3 import cyber
from cyber.python.cyber_py3 import cyber_time

from modules.drivers.proto import pointcloud_pb2 

class CompensatorPointCloudPublisher(object):
    def __init__(self, node):
        self.pointcloud = pointcloud_pb2.PointCloud()
        self.compensator_pointcloud_pub = node.create_writer('/apollo/sensor/lidar16/compensator/PointCloud2', pointcloud_pb2.PointCloud)        

    def pointcloud_callback(self, data):
        """
        New message received
        """
        
        self.pointcloud.CopyFrom(data)
        now = cyber_time.Time.now().to_sec()        
        self.pointcloud.header.timestamp_sec = now
        self.pointcloud.measurement_time = now

    def publish_compensator_pointcloud(self):
        self.compensator_pointcloud_pub.write(self.pointcloud)

    def shutdown(self):
        """
        shutdown rosnode
        """
        self.terminating = True
        #self.logger.info("Shutting Down...")
        time.sleep(0.2)

def main():
    """
    Main rosnode
    """
    node = cyber.Node('lidar_compensator_publisher')
    lidar_compensator = CompensatorPointCloudPublisher(node)
    node.create_reader('/apollo/sensor/lidar16/PointCloud2', pointcloud_pb2.PointCloud, lidar_compensator.pointcloud_callback)
    while not cyber.is_shutdown():
        now = cyber_time.Time.now().to_sec()
        lidar_compensator.publish_compensator_pointcloud()
        sleep_time = 0.1 - (cyber_time.Time.now().to_sec() - now)
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == '__main__':
    cyber.init()
    main()
    cyber.shutdown()
