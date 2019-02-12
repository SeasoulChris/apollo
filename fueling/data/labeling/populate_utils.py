#!/usr/bin/env python
import os
import subprocess
import sys
import time

#import cv2
import math
import numpy as np
import yaml
from copy import deepcopy
from google.protobuf.json_format import MessageToJson

from cyber_py import cyber
from cyber_py import record
import fueling.common.s3_utils as s3_utils

from modules.data.proto import frame_pb2
from modules.common.proto.geometry_pb2 import Point3D
from modules.common.proto.geometry_pb2 import Quaternion
from modules.drivers.proto.conti_radar_pb2 import ContiRadar
from modules.drivers.proto.pointcloud_pb2 import PointCloud
from modules.drivers.proto.sensor_image_pb2 import CompressedImage
from modules.drivers.proto.sensor_image_pb2 import Image
from modules.localization.proto.localization_pb2 import LocalizationEstimate

# Map channels to processing functions
g_channel_process_map = {}

sensor_params = {
    'lidar_channel': '/apollo/sensor/lidar128/compensator/PointCloud2',
    'lidar_extrinsics': '/modules/calibration/mkz6/velodyne_params/velodyne128_novatel_extrinsics.yaml',

    'front6mm_channel': '/apollo/sensor/camera/front_6mm/image/compressed',
    'front6mm_intrinsics': '/modules/calibration/mkz6/camera_params/front_6mm_intrinsics.yaml',
    'front6mm_extrinsics': 'modules/calibration/mkz6/camera_params/front_6mm_extrinsics.yaml',

    'front12mm_channel': '/apollo/sensor/camera/front_12mm/image/compressed',
    'front12mm_intrinsics': '/modules/calibration/mkz6/camera_params/front_12mm_intrinsics.yaml',
    'front12mm_extrinsics': '/modules/calibration/mkz6/camera_params/front_12mm_extrinsics.yaml',

    'left_fisheye_channel': '/apollo/sensor/camera/left_fisheye/image/compressed',
    'left_fisheye_intrinsics': '/modules/calibration/mkz6/camera_params/left_fisheye_intrinsics.yaml',
    'left_fisheye_extrinsics': '/modules/calibration/mkz6/camera_params/left_fisheye_velodyne128_extrinsics.yaml',

    'right_fisheye_channel': '/apollo/sensor/camera/right_fisheye/image/compressed',
    'right_fisheye_intrinsics': '/modules/calibration/mkz6/camera_params/right_fisheye_intrinsics.yaml',
    'right_fisheye_extrinsics': '/modules/calibration/mkz6/camera_params/right_fisheye_velodyne128_extrinsics.yaml',

    'rear6mm_channel': '/apollo/sensor/camera/rear_6mm/image/compressed',
    'rear6mm_intrinsics': '/modules/calibration/mkz6/camera_params/rear_6mm_intrinsics.yaml',
    'rear6mm_extrinsics': '/modules/calibration/mkz6/camera_params/rear_6mm_extrinsics.yaml',

    'pose_channel': '/apollo/localization/pose',
    
    'radar_front_channel': '/apollo/sensor/radar/front',
    'radar_front_extrinsics': '/modules/calibration/mkz6/radar_params/radar_front_extrinsics.yaml',

    'radar_rear_channel': '/apollo/sensor/radar/rear',
    'radar_rear_extrinsics': '/modules/calibration/mkz6/radar_params/radar_rear_extrinsics.yaml',

    'image_url': 'https://s3-us-west-1.amazonaws.com/scale-labeling'
}

def run_shell_script(command):
    """Simple wrapper to run shell command"""
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.communicate()
    return p.returncode

def load_yaml_settings(yaml_file_name):
    """Load settings from YAML config file."""
    if yaml_file_name is None:
        return None
    yaml_file_name = os.path.join(s3_utils.S3MountPath, yaml_file_name)
    yaml_file = open(yaml_file_name)
    return yaml.safe_load(yaml_file)

def dump_img_bin(data, output_dir, frame_id, channel):
    """Dump image bytes to binary file."""
    if not os.path.exists(output_dir):
        run_shell_script('sudo mkdir -p {} -m 755'.format(output_dir))
    with open('{}/image_bin-{}_{}'.format(output_dir, frame_id, channel), 'wb') as bin_file:
        bin_file.write(data)

def point3d_to_matrix(P):
    """Convert a 3-items array to 4*1 matrix."""
    mat = np.zeros(shape=(4,1), dtype=float) 
    mat = np.array([[P.x],[P.y],[P.z],[1]])
    return mat

def quaternion_to_roation(Q):
    """Convert quaternion vector to 3x3 rotation matrix."""
    rotation_mat = np.zeros(shape=(3,3), dtype=float)
    rotation_mat[0][0] = Q.qw**2 + Q.qx**2 - Q.qy**2 - Q.qz**2
    rotation_mat[0][1] = 2 * (Q.qx*Q.qy - Q.qw*Q.qz)
    rotation_mat[0][2] = 2 * (Q.qx*Q.qz + Q.qw*Q.qy)
    rotation_mat[1][0] = 2 * (Q.qx*Q.qy + Q.qw*Q.qz)
    rotation_mat[1][1] = Q.qw**2 - Q.qx**2 + Q.qy**2 - Q.qz**2
    rotation_mat[1][2] = 2 * (Q.qy*Q.qz - Q.qw*Q.qx)
    rotation_mat[2][0] = 2 * (Q.qx*Q.qz - Q.qw*Q.qy)    
    rotation_mat[2][1] = 2 * (Q.qy*Q.qz + Q.qw*Q.qx)    
    rotation_mat[2][2] = Q.qw**2 - Q.qx**2 - Q.qy**2 + Q.qz**2
    return rotation_mat

def rotation_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion vector."""
    q = Quaternion()
    q.qx = np.absolute(np.sqrt(1+R[0][0]-R[1][1]-R[2][2])) * \
        np.sign(R[2][1]-R[1][2]) * 0.5
    q.qy = np.absolute(np.sqrt(1-R[0][0]+R[1][1]-R[2][2])) * \
        np.sign(R[0][2]-R[2][0]) * 0.5 
    q.qz = np.absolute(np.sqrt(1-R[0][0]-R[1][1]+R[2][2])) * \
        np.sign(R[1][0]-R[0][1]) * 0.5
    q.qw = np.sqrt(1 - q.qx * q.qx - q.qy * q.qy - q.qz * q.qz)
    return q

def generate_transform(Q, D):
    """Generate a matrix with rotation and deviation/translation."""
    tranform = np.zeros(shape=(4,4), dtype=float)
    tranform[0][0] = Q.qw**2 + Q.qx**2 - Q.qy**2 - Q.qz**2
    tranform[0][1] = 2 * (Q.qx*Q.qy - Q.qw*Q.qz)
    tranform[0][2] = 2 * (Q.qx*Q.qz + Q.qw*Q.qy)
    tranform[1][0] = 2 * (Q.qx*Q.qy + Q.qw*Q.qz)
    tranform[1][1] = Q.qw**2 - Q.qx**2 + Q.qy**2 - Q.qz**2
    tranform[1][2] = 2 * (Q.qy*Q.qz - Q.qw*Q.qx)
    tranform[2][0] = 2 * (Q.qx*Q.qz - Q.qw*Q.qy)
    tranform[2][1] = 2 * (Q.qy*Q.qz + Q.qw*Q.qx)
    tranform[2][2] = Q.qw**2 - Q.qx**2 - Q.qy**2 + Q.qz**2
    tranform[0][3] = D.x 
    tranform[1][3] = D.y 
    tranform[2][3] = D.z
    tranform[3] = [0,0,0,1]
    return tranform

def get_rotation_from_tranform(T):
    """Extract rotation matrix out from transform matrix."""
    rotation = np.zeros(shape=(3,3), dtype=float)
    rotation[0][0] = T[0][0]
    rotation[0][1] = T[0][1]
    rotation[0][2] = T[0][2]
    rotation[1][0] = T[1][0]
    rotation[1][1] = T[1][1]
    rotation[1][2] = T[1][2]
    rotation[2][0] = T[2][0]
    rotation[2][1] = T[2][1]
    rotation[2][2] = T[2][2]
    return rotation

def transform_coordinate(P, T):
    """Transform coordinate system according to rotation and translation."""
    point_mat = point3d_to_matrix(P)
    #point_rotation = np.matmul(T, point_mat)
    point_mat = np.dot(T, point_mat)
    P.x = point_mat[0][0]
    P.y = point_mat[1][0]
    P.z = point_mat[2][0]

def multiply_quaternion(Q1, Q2):
    """Multiple two quaternions. Q1 is the rotation applied AFTER Q2."""
    q = Quaternion()
    q.qw = Q1.qw*Q2.qw - Q1.qx*Q2.qx - Q1.qy*Q2.qy - Q1.qz*Q2.qz
    q.qx = Q1.qw*Q2.qx + Q1.qx*Q2.qw + Q1.qy*Q2.qz - Q1.qz*Q2.qy
    q.qy = Q1.qw*Q2.qy - Q1.qx*Q2.qy + Q1.qy*Q2.qw + Q1.qz*Q2.qx
    q.qz = Q1.qw*Q2.qz + Q1.qx*Q2.qy - Q1.qy*Q2.qx + Q1.qz*Q2.qw
    return q

def get_world_coordinate(T, pose):
    """Get world coordinate by using transform matrix (imu pose)"""
    pose_transform = generate_transform(pose._orientation, pose._position)
    T = np.dot(pose_transform, T)
    return T

def convert_to_world_coordinate(P, T, stationary_pole):
    """
    Convert to world coordinate by two steps:
    1. from imu to world by using transform matrix (imu pose)
    2. every point substract by the original point to match the visualizer
    """
    transform_coordinate(P, T)
    P.x -= stationary_pole[0]
    P.y -= stationary_pole[1]
    P.z -= stationary_pole[2]

def pose_to_stationary_pole(pose_message):
    _channel, message, _type, _timestamp = pose_message
    localization = LocalizationEstimate()
    localization.ParseFromString(message)
    position = localization.pose.position
    return (position.x, position.y, position.z)

class Sensor(object):
    """Sensor class representing various of sensors."""
    def __init__(self, channel, intrinsics, extrinsics, stationary_pole):
        """Constructor."""
        self._channel = channel
        self._intrinsics = load_yaml_settings(intrinsics)
        self._extrinsics = load_yaml_settings(extrinsics)
        self._stationary_pole = stationary_pole
        self.initialize_transform()
        g_channel_process_map[self._channel] = self
 
    def process(self, message, frame, pose):
        """Processing function."""
        pass

    def initialize_transform(self):
        if self._extrinsics is None:
            return
        q = Quaternion()
        q.qw = self._extrinsics['transform']['rotation']['w']
        q.qx = self._extrinsics['transform']['rotation']['x']
        q.qy = self._extrinsics['transform']['rotation']['y']
        q.qz = self._extrinsics['transform']['rotation']['z']
        d = Point3D()
        d.x = self._extrinsics['transform']['translation']['x']
        d.y = self._extrinsics['transform']['translation']['y']
        d.z = self._extrinsics['transform']['translation']['z']
        self._transform = generate_transform(q, d)

    def add_transform(self, transform):
        self._transform = np.dot(transform, self._transform)

class PointCloudSensor(Sensor):
    """Lidar sensor that hold pointcloud data."""
    def __init__(self, channel, intrinsics, extrinsics, stationary_pole):
        """Initialization."""
        super(PointCloudSensor, self).__init__(channel, intrinsics, extrinsics, stationary_pole)

    def process(self, message, frame, pose):
        """Process PointCloud message."""
        point_cloud = PointCloud()
        point_cloud.ParseFromString(message)
        transform = get_world_coordinate(self._transform, pose)
        for point in point_cloud.point:
            convert_to_world_coordinate(point, transform, self._stationary_pole)
            vector4 = frame_pb2.Vector4()
            vector4.x = point.x
            vector4.y = point.y
            vector4.z = point.z
            vector4.i = point.intensity
            point_frame = frame.points.add()
            point_frame.CopyFrom(vector4)

        point = Point3D()
        point.x = 0; point.y = 0; point.z = 0
        transform = get_world_coordinate(self._transform, pose)
        convert_to_world_coordinate(point, transform, self._stationary_pole)
        frame.device_position.x = point.x
        frame.device_position.y = point.y
        frame.device_position.z = point.z
        rotation = get_rotation_from_tranform(transform)
        q = rotation_to_quaternion(rotation)
        # TODO: either apply it to all or do not apply it
        #q = apply_scale_rotation(q)
        frame.device_heading.x = q.qx
        frame.device_heading.y = q.qy
        frame.device_heading.z = q.qz
        frame.device_heading.w = q.qw

class RadarSensor(Sensor):
    """Radar sensor that hold radar data."""
    def __init__(self, channel, intrinsics, extrinsics, transforms, type, stationary_pole):
        """Initialization."""
        super(RadarSensor, self).__init__(channel, intrinsics, extrinsics, stationary_pole)
        self._type = type
        for T in transforms:
            self.add_transform(T)

    def process(self, message, frame, pose):
        """Processing radar message."""
        radar = ContiRadar()
        radar.ParseFromString(message)
        transform = get_world_coordinate(self._transform, pose)
        for point in radar.contiobs:
            point3d = Point3D()
            point3d.x = point.longitude_dist
            point3d.y = point.lateral_dist
            point3d.z = 0
            convert_to_world_coordinate(point3d, transform, self._stationary_pole)
            radar_point = frame_pb2.RadarPoint()
            radar_point.type = self._type 
            radar_point.position.x = point3d.x
            radar_point.position.y = point3d.y
            radar_point.position.z = point3d.z
            #radar_point.position.z = 0
            point3d.x = point.longitude_dist + point.longitude_vel
            point3d.y = point.lateral_dist + point.lateral_vel
            point3d.z = 0
            convert_to_world_coordinate(point3d, transform, self._stationary_pole)
            radar_point.direction.x = point3d.x - radar_point.position.x
            radar_point.direction.y = point3d.y - radar_point.position.y
            radar_point.direction.z = point3d.z - radar_point.position.z
            #radar_point.direction.z = 0
            radar_frame = frame.radar_points.add()
            radar_frame.CopyFrom(radar_point)

class ImageSensor(Sensor):
    """Image sensor that hold camera data."""
    def __init__(self, channel, intrinsics, extrinsics, task_dir, transforms, stationary_pole):
        """Initialization."""
        super(ImageSensor, self).__init__(channel, intrinsics, extrinsics, stationary_pole)
        self._task_dir = task_dir
        self._frame_id = 0
        for T in transforms:
            self.add_transform(T)

    def process(self, message, frame, pose):
        """Processing image message."""
        image = CompressedImage()
        image.ParseFromString(message)
        camera_image = frame_pb2.CameraImage()
        camera_image.timestamp = image.header.timestamp_sec
        dump_img_bin(image.data, 
            os.path.join(self._task_dir, 'images'), 
            self._frame_id, 
            self.get_image_name())
        camera_image.image_url = '{}/{}/images/pic-{}_{}.jpg'.format(
            sensor_params['image_url'], 
            os.path.basename(self._task_dir),
            self._frame_id,
            self.get_image_name())
        camera_image.k1 = self._intrinsics['D'][0]
        camera_image.k2 = self._intrinsics['D'][1]
        camera_image.k3 = self._intrinsics['D'][4]
        camera_image.p1 = self._intrinsics['D'][2]
        camera_image.p2 = self._intrinsics['D'][3]
        camera_image.skew = self._intrinsics['K'][1]
        camera_image.fx = self._intrinsics['K'][0]
        camera_image.fy = self._intrinsics['K'][4]
        camera_image.cx = self._intrinsics['K'][2]
        camera_image.cy = self._intrinsics['K'][5]
        point = Point3D()
        point.x = 0; point.y = 0; point.z = 0
        transform = get_world_coordinate(self._transform, pose)
        convert_to_world_coordinate(point, transform, self._stationary_pole)
        camera_image.position.x = point.x
        camera_image.position.y = point.y
        camera_image.position.z = point.z
        rotation = get_rotation_from_tranform(transform)
        q = rotation_to_quaternion(rotation)
        # TODO: either apply it to all or do not apply it
        #q = apply_scale_rotation(q)
        camera_image.heading.x = q.qx
        camera_image.heading.y = q.qy
        camera_image.heading.z = q.qz
        camera_image.heading.w = q.qw
        camera_image.channel = self._channel
        image_frame = frame.images.add()
        image_frame.CopyFrom(camera_image)
    
    def get_image_name(self):
        """A nasty way to get image name from map."""
        for name in sensor_params:
            if sensor_params[name] == self._channel:
                return name

class GpsSensor(Sensor):
    """GPS sensor that hold pose data."""
    def __init__(self, channel, intrinsics, extrinsics, stationary_pole):
        """Initialization."""
        super(GpsSensor, self).__init__(channel, intrinsics, extrinsics, None)
        
    def process(self, message, frame, pose):
        """Process Pose message."""
        localization = LocalizationEstimate()
        localization.ParseFromString(message)

        self._position = localization.pose.position
        self._orientation = localization.pose.orientation

        gps_pose = frame_pb2.GPSPose()
        gps_pose.lat = localization.pose.position.x
        gps_pose.lon = localization.pose.position.y
        gps_pose.bearing = localization.pose.orientation.qw
        gps_pose.x = localization.pose.position.x
        gps_pose.y = localization.pose.position.y
        gps_pose.z = localization.pose.position.z
        gps_pose.qw = localization.pose.orientation.qw
        gps_pose.qx = localization.pose.orientation.qx
        gps_pose.qy = localization.pose.orientation.qy
        gps_pose.qz = localization.pose.orientation.qz
        frame.device_gps_pose.CopyFrom(gps_pose)

class FramePopulator:
    """Extract sensors data from record file, and populate to JSON."""
    def __init__(self, task_dir, pose_message):     
        stationary_pole = pose_to_stationary_pole(pose_message) 

        self._task_dir = os.path.join(s3_utils.S3MountPath, task_dir)
        if not os.path.exists(self._task_dir):
            run_shell_script('sudo mkdir -p {} -m 755'.format(self._task_dir))

        self._gps_pose = GpsSensor(sensor_params['pose_channel'], None, None, None)
        self._pointcloud_128 = PointCloudSensor(
            channel=sensor_params['lidar_channel'],
            intrinsics=None,
            extrinsics=sensor_params['lidar_extrinsics'],
            stationary_pole=stationary_pole
        )
        self._image_front_6mm = ImageSensor(
            channel=sensor_params['front6mm_channel'],
            intrinsics=sensor_params['front6mm_intrinsics'],
            extrinsics=sensor_params['front6mm_extrinsics'],
            task_dir=self._task_dir,
            transforms=[self._pointcloud_128._transform],
            stationary_pole=stationary_pole)
        self._image_front_12mm = ImageSensor(
            channel=sensor_params['front12mm_channel'],
            intrinsics=sensor_params['front12mm_intrinsics'],
            extrinsics=sensor_params['front12mm_extrinsics'],
            task_dir=self._task_dir,
            transforms=[self._pointcloud_128._transform],
            stationary_pole=stationary_pole)
        self._image_left_fisheye = ImageSensor(
            channel=sensor_params['left_fisheye_channel'],
            intrinsics=sensor_params['left_fisheye_intrinsics'],
            extrinsics=sensor_params['left_fisheye_extrinsics'],
            task_dir=self._task_dir,
            transforms=[self._pointcloud_128._transform],
            stationary_pole=stationary_pole)
        self._image_right_fisheye = ImageSensor(
            channel=sensor_params['right_fisheye_channel'],
            intrinsics=sensor_params['right_fisheye_intrinsics'],
            extrinsics=sensor_params['right_fisheye_extrinsics'],
            task_dir=self._task_dir,
            transforms=[self._pointcloud_128._transform],
            stationary_pole=stationary_pole)
        self._image_rear = ImageSensor(
            channel=sensor_params['rear6mm_channel'],
            intrinsics=sensor_params['rear6mm_intrinsics'],
            extrinsics=sensor_params['rear6mm_extrinsics'],
            task_dir=self._task_dir,
            transforms=[self._pointcloud_128._transform],
            stationary_pole=stationary_pole)
        self._radar_front = RadarSensor(
            channel=sensor_params['radar_front_channel'],
            intrinsics=None,
            extrinsics=sensor_params['radar_front_extrinsics'],
            transforms=[self._pointcloud_128._transform],
            type=frame_pb2.RadarPoint.FRONT,
            stationary_pole=stationary_pole)
        self._radar_rear = RadarSensor(
            channel=sensor_params['radar_rear_channel'],
            intrinsics=None,
            extrinsics=sensor_params['radar_rear_extrinsics'],
            transforms=[self._pointcloud_128._transform],
            type=frame_pb2.RadarPoint.REAR,
            stationary_pole=stationary_pole)

    def construct_frames(self, msgs):
        """Construct the frames by using given messages."""
        frame = frame_pb2.Frame()
        lidar_msg = next(x for x in msgs if x.topic == sensor_params['lidar_channel'])

        msgs.sort(key=lambda x: x.topic==sensor_params['pose_channel'], reverse=True)

        for msg in msgs:
            channel, message, _type, _timestamp = msg
            if isinstance(g_channel_process_map[channel], ImageSensor):
                g_channel_process_map[channel]._frame_id = lidar_msg.timestamp
            g_channel_process_map[channel].process(message, frame, g_channel_process_map[sensor_params['pose_channel']])

        frame.timestamp = lidar_msg.timestamp
        frame_dir = os.path.join(self._task_dir, 'frames')
        if not os.path.exists(frame_dir):
            run_shell_script('sudo mkdir -p {} -m 755'.format(frame_dir))
        
        file_name = os.path.join(frame_dir, 'frame-{}.json'.format(lidar_msg.timestamp))
        jsonObj = MessageToJson(frame, False, True)
        with open(file_name, 'w') as outfile:
            outfile.write(jsonObj)
  
