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

from modules.localization.proto import localization_pb2
from modules.transform.proto import transform_pb2

class PoseTFPublisher(object):
    def __init__(self, node):
        #/pose
        self.localization = localization_pb2.LocalizationEstimate()
        self.localization_pub = node.create_writer('/apollo/localization/pose', localization_pb2.LocalizationEstimate) 
        self.sequence_num = 0
        self.terminating = False
        #/tf
        self.tf = transform_pb2.TransformStampeds()
        self.tf_arrary = transform_pb2.TransformStamped()
        self.tf_pub = node.create_writer('/tf', transform_pb2.TransformStampeds) 

    def publish_localization(self):
        now = cyber_time.Time.now().to_sec()
        self.localization.header.timestamp_sec = now
        self.localization.header.module_name = "localization"
        self.localization.header.sequence_num = self.sequence_num
        self.sequence_num = self.sequence_num + 1
        self.localization.pose.position.x = 437908.896
        self.localization.pose.position.y = 4433029.036
        self.localization.pose.position.z = 0.0
        self.localization.pose.orientation.qx = 0.0
        self.localization.pose.orientation.qy = 0.0
        self.localization.pose.orientation.qz = 0.76518
        self.localization.pose.orientation.qw = 0.64392
        self.localization.pose.linear_velocity.x = 0.0
        self.localization.pose.linear_velocity.y = 0.0
        self.localization.pose.linear_velocity.z = 0.0
        self.localization_pub.write(self.localization)



    def publish_tf(self):
        now = cyber_time.Time.now().to_sec()
        tf = transform_pb2.TransformStampeds()
        self.tf_arrary.header.timestamp_sec = now
        self.tf_arrary.header.frame_id = 'world'    
        self.tf_arrary.child_frame_id = 'localization'
        self.tf_arrary.transform.translation.x = 437908.896   
        self.tf_arrary.transform.translation.y = 4433029.036
        self.tf_arrary.transform.translation.z = 0.0
        self.tf_arrary.transform.rotation.qx = 0.0
        self.tf_arrary.transform.rotation.qy = 0.0
        self.tf_arrary.transform.rotation.qz = 0.76518
        self.tf_arrary.transform.rotation.qw = 0.64392
        tf.transforms.append(self.tf_arrary)
        self.tf_pub.write(tf)

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
    node_localization = cyber.Node('localization_publisher')
    node_tf = cyber.Node('tf_publisher')
    localization = PoseTFPublisher(node_localization)
    tf =  PoseTFPublisher(node_tf)
    #node.create_reader('/apollo/localization/pose', localization_pb2.LocalizationEstimate, odom.localization_callback)
    while not cyber.is_shutdown():
        now = cyber_time.Time.now().to_sec()
        localization.publish_localization()
        tf.publish_tf()
        sleep_time = 0.01 - (cyber_time.Time.now().to_sec() - now)
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == '__main__':
    cyber.init()
    main()
    cyber.shutdown()
