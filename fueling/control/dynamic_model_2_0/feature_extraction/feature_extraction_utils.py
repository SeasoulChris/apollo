#!/usr/bin/python

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np


def chassis_msg_to_data(chassis):
    """Extract numpy array from proto"""
    return np.array([
               chassis.speed_mps,  # 14 speed
               chassis.throttle_percentage / 100,  # 15 throttle
               chassis.brake_percentage / 100,  # 16 brake
               chassis.steering_percentage / 100,  # 17
               chassis.driving_mode,  # 18
               chassis.gear_location,  # 22
           ])


def localization_msg_to_data(pose):
    """Extract numpy array from proto"""
    pose = pose.pose
    return np.array([
               pose.heading,  # 0
               pose.orientation.qx,  # 1
               pose.orientation.qy,  # 2
               pose.orientation.qz,  # 3
               pose.orientation.qw,  # 4
               pose.linear_velocity.x,  # 5
               pose.linear_velocity.y,  # 6
               pose.linear_velocity.z,  # 7
               pose.linear_acceleration.x,  # 8
               pose.linear_acceleration.y,  # 9
               pose.linear_acceleration.z,  # 10
               pose.angular_velocity.x,  # 11
               pose.angular_velocity.y,  # 12
               pose.angular_velocity.z,  # 13
               pose.position.x,  # 19
               pose.position.y,  # 20
               pose.position.z,  # 21
           ])


def feature_combine(chassis_data, pose_data):
    """Compbine chassis and localization data"""
    return np.hstack((pose_data[:14], chassis_data[:5], pose_data[14:], chassis_data[5]))


def interpolate_pose(pose_left, pose_right, time_in_between):
    """Get interpolation of two pose protos"""

    def liner_interp(time_left, time_right, value_left, value_right, time_in_between):
        """Liner interpolation"""
        func = interp1d([time_left, time_right], [value_left, value_right], kind='linear')
        return func([time_in_between])[0]

    def rotation_interp(time_left, time_right, rotation_left, rotation_right, time_in_between):
        """Rotation interpolation, rotation is in Quaternion format"""
        times = [time_left, time_right]
        rotations = R.from_quat([rotation_left, rotation_right])
        slerp = Slerp(times, rotations)
        return slerp(time_in_between).as_quat()

    # Do rotation interpolation for orientation, and liner for everything else"""
    time_left, data_left = pose_left.header.timestamp_sec, localization_msg_to_data(pose_left)
    time_right, data_right = pose_right.header.timestamp_sec, localization_msg_to_data(pose_right)
    interp_pose = np.array([0.0] * len(data_left))
    rotation_indexes = [1, 2, 3, 4]
    rot_left = [data_left[x] for x in rotation_indexes] 
    rot_right = [data_right[x] for x in rotation_indexes] 
    rotation = rotation_interp(time_left, time_right, rot_left, rot_right, time_in_between)
    for index in range(len(data_left)):
        if index not in rotation_indexes:
            interp_pose[index] = liner_interp(
                time_left, time_right, data_left[index], data_right[index], time_in_between)
        else:
            interp_pose[index] = rotation[index - rotation_indexes[0]]
    return interp_pose
