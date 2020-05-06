#!/usr/bin/env python

from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config as config


class InterPolationMessage(object):
    """Encapsulation of interpolation messages"""

    def __init__(self, chasis_msg=None, left_pose_msg=None, right_pose_msg=None):
        """Init, the messages here are message protos"""
        self.chasis_msg = chasis_msg
        self.left_pose_msg = left_pose_msg
        self.right_pose_msg = right_pose_msg 
        self.interpolate_pose = None


    def add_pose(self, pose_msg):
        """Add a pose to the current object, need to check if it's left or right""" 
        if not self.chasis_msg or not pose_msg:
            return
        if pose_msg.header.timestamp_sec <= self.chasis_msg.header.timestamp_sec:
            self.left_pose_msg = pose_msg
        elif not self.right_pose_msg:
            self.right_pose_msg = pose_msg


    def is_valid(self):
        """Check if the current object is valid"""
        if not self.chasis_msg or not self.left_pose_msg or not self.right_pose_msg:
            return False
        if (self.chasis_msg.header.timestamp_sec - self.left_pose_msg.header.timestamp_sec
                > config['MAX_POSE_DELTA'] or
            self.right_pose_msg.header.timestamp_sec - self.chasis_msg.header.timestamp_sec
                > config['MAX_POSE_DELTA']):
            return False
        return True


    def do_interpolation(self):
        """Actually do the interpolation"""
        # TODO(longtao): implement in next iteration
        return
