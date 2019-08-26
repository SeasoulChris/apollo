#!/usr/bin/env python

"""Utils classes and functions to support populating frames"""

import gc
import math
import os

from google.protobuf.json_format import MessageToJson
from pyquaternion import Quaternion as PyQuaternion
import colored_glog as glog
import cv2
import numpy as np
import pypcd
import yaml

from cyber_py import record
from modules.data.proto import frame_pb2
from modules.common.proto.geometry_pb2 import Point3D
from modules.common.proto.geometry_pb2 import PointENU
from modules.common.proto.geometry_pb2 import Quaternion
from modules.drivers.proto.conti_radar_pb2 import ContiRadar
from modules.drivers.proto.pointcloud_pb2 import PointCloud
from modules.localization.proto.localization_pb2 import LocalizationEstimate

import fueling.common.file_utils as file_utils
import fueling.common.storage.bos_client as bos_client
import fueling.streaming.streaming_utils as streaming_utils

# Map channels to processing functions
CHANNEL_PROCESS_MAP = {}

# Set precision explicitly
np.set_printoptions(precision=17)

SENSOR_PARAMS = {
    'lidar_channel':
    '/apollo/sensor/lidar128/compensator/PointCloud2',
    'lidar_extrinsics':
    'modules/calibration/mkz6/velodyne_params/velodyne128_novatel_extrinsics.yaml',

    'front6mm_channel':
    '/apollo/sensor/camera/front_6mm/image/compressed',
    'front6mm_intrinsics':
    'modules/calibration/mkz6/camera_params/front_6mm_intrinsics.yaml',
    'front6mm_extrinsics':
    'modules/calibration/mkz6/camera_params/front_6mm_extrinsics.yaml',

    'front12mm_channel':
    '/apollo/sensor/camera/front_12mm/image/compressed',
    'front12mm_intrinsics':
    'modules/calibration/mkz6/camera_params/front_12mm_intrinsics.yaml',
    'front12mm_extrinsics':
    'modules/calibration/mkz6/camera_params/front_12mm_extrinsics.yaml',

    'left_fisheye_channel':
    '/apollo/sensor/camera/left_fisheye/image/compressed',
    'left_fisheye_intrinsics':
    'modules/calibration/mkz6/camera_params/left_fisheye_intrinsics.yaml',
    'left_fisheye_extrinsics':
    'modules/calibration/mkz6/camera_params/left_fisheye_velodyne128_extrinsics.yaml',

    'right_fisheye_channel':
    '/apollo/sensor/camera/right_fisheye/image/compressed',
    'right_fisheye_intrinsics':
    'modules/calibration/mkz6/camera_params/right_fisheye_intrinsics.yaml',
    'right_fisheye_extrinsics':
    'modules/calibration/mkz6/camera_params/right_fisheye_velodyne128_extrinsics.yaml',

    'rear6mm_channel':
    '/apollo/sensor/camera/rear_6mm/image/compressed',
    'rear6mm_intrinsics':
    'modules/calibration/mkz6/camera_params/rear_6mm_intrinsics.yaml',
    'rear6mm_extrinsics':
    'modules/calibration/mkz6/camera_params/rear_6mm_extrinsics.yaml',

    'pose_channel':
    '/apollo/localization/pose',

    'radar_front_channel':
    '/apollo/sensor/radar/front',
    'radar_front_extrinsics':
    'modules/calibration/mkz6/radar_params/radar_front_extrinsics.yaml',

    'radar_rear_channel':
    '/apollo/sensor/radar/rear',
    'radar_rear_extrinsics':
    'modules/calibration/mkz6/radar_params/radar_rear_extrinsics.yaml',

    'image_url':
    'https://s3-us-west-1.amazonaws.com/scale-labeling/images'
}

def load_yaml_settings(yaml_file_name):
    """Load settings from YAML config file."""
    if yaml_file_name is None:
        return None
    yaml_file_name = bos_client.abs_path(yaml_file_name)
    yaml_file = open(yaml_file_name)
    return yaml.safe_load(yaml_file)

def dump_img_name(output_dir, image_name):
    """Write image file name only"""
    file_utils.makedirs(output_dir)
    image_file_path = os.path.join(output_dir, image_name)
    if not os.path.exists(image_file_path):
        os.mknod(image_file_path)

def point3d_to_matrix(point):
    """Convert a 3-items array to 4*1 matrix."""
    mat = np.zeros(shape=(4, 1), dtype=float)
    mat = np.array([[point.x], [point.y], [point.z], [1]])
    return mat

def quaternion_to_roation(qtn):
    """Convert quaternion vector to 3x3 rotation matrix."""
    rotation_mat = np.zeros(shape=(3, 3), dtype=float)
    rotation_mat[0][0] = qtn.qw**2 + qtn.qx**2 - qtn.qy**2 - qtn.qz**2
    rotation_mat[0][1] = 2 * (qtn.qx*qtn.qy - qtn.qw*qtn.qz)
    rotation_mat[0][2] = 2 * (qtn.qx*qtn.qz + qtn.qw*qtn.qy)
    rotation_mat[1][0] = 2 * (qtn.qx*qtn.qy + qtn.qw*qtn.qz)
    rotation_mat[1][1] = qtn.qw**2 - qtn.qx**2 + qtn.qy**2 - qtn.qz**2
    rotation_mat[1][2] = 2 * (qtn.qy*qtn.qz - qtn.qw*qtn.qx)
    rotation_mat[2][0] = 2 * (qtn.qx*qtn.qz - qtn.qw*qtn.qy)
    rotation_mat[2][1] = 2 * (qtn.qy*qtn.qz + qtn.qw*qtn.qx)
    rotation_mat[2][2] = qtn.qw**2 - qtn.qx**2 - qtn.qy**2 + qtn.qz**2
    return rotation_mat

def rotation_to_quaternion(rot):
    """Convert 3x3 rottation matrix to quaternion vector."""
    qtn = Quaternion()
    qtn.qx = np.absolute(np.sqrt(1+rot[0][0]-rot[1][1]-rot[2][2])) * \
        np.sign(rot[2][1]-rot[1][2]) * 0.5
    qtn.qy = np.absolute(np.sqrt(1-rot[0][0]+rot[1][1]-rot[2][2])) * \
        np.sign(rot[0][2]-rot[2][0]) * 0.5
    qtn.qz = np.absolute(np.sqrt(1-rot[0][0]-rot[1][1]+rot[2][2])) * \
        np.sign(rot[1][0]-rot[0][1]) * 0.5
    qtn.qw = np.sqrt(1 - qtn.qx * qtn.qx - qtn.qy * qtn.qy - qtn.qz * qtn.qz)
    return qtn

def generate_transform(qtn, dev):
    """Generate a matrix with rotation and deviation/translation."""
    tranform = np.zeros(shape=(4, 4), dtype=float)
    tranform[0][0] = qtn.qw**2 + qtn.qx**2 - qtn.qy**2 - qtn.qz**2
    tranform[0][1] = 2 * (qtn.qx*qtn.qy - qtn.qw*qtn.qz)
    tranform[0][2] = 2 * (qtn.qx*qtn.qz + qtn.qw*qtn.qy)
    tranform[1][0] = 2 * (qtn.qx*qtn.qy + qtn.qw*qtn.qz)
    tranform[1][1] = qtn.qw**2 - qtn.qx**2 + qtn.qy**2 - qtn.qz**2
    tranform[1][2] = 2 * (qtn.qy*qtn.qz - qtn.qw*qtn.qx)
    tranform[2][0] = 2 * (qtn.qx*qtn.qz - qtn.qw*qtn.qy)
    tranform[2][1] = 2 * (qtn.qy*qtn.qz + qtn.qw*qtn.qx)
    tranform[2][2] = qtn.qw**2 - qtn.qx**2 - qtn.qy**2 + qtn.qz**2
    tranform[0][3] = dev.x
    tranform[1][3] = dev.y
    tranform[2][3] = dev.z
    tranform[3] = [0, 0, 0, 1]
    return tranform

def get_rotation_from_tranform(transform):
    """Extract rotation matrix out from transform matrix."""
    rotation = np.zeros(shape=(3, 3), dtype=float)
    rotation[0][0] = transform[0][0]
    rotation[0][1] = transform[0][1]
    rotation[0][2] = transform[0][2]
    rotation[1][0] = transform[1][0]
    rotation[1][1] = transform[1][1]
    rotation[1][2] = transform[1][2]
    rotation[2][0] = transform[2][0]
    rotation[2][1] = transform[2][1]
    rotation[2][2] = transform[2][2]
    return rotation

def transform_coordinate(point, transform):
    """Transform coordinate system according to rotation and translation."""
    point_mat = point3d_to_matrix(point)
    point_mat = np.dot(transform, point_mat)
    trans_point = Point3D()
    trans_point.x = point_mat[0][0]
    trans_point.y = point_mat[1][0]
    trans_point.z = point_mat[2][0]
    return trans_point

def multiply_quaternion(qtn1, qtn2):
    """Multiple two quaternions. qtn1 is the rotation applied AFTER qtn2."""
    qtn = Quaternion()
    qtn.qw = qtn1.qw*qtn2.qw - qtn1.qx*qtn2.qx - qtn1.qy*qtn2.qy - qtn1.qz*qtn2.qz
    qtn.qx = qtn1.qw*qtn2.qx + qtn1.qx*qtn2.qw + qtn1.qy*qtn2.qz - qtn1.qz*qtn2.qy
    qtn.qy = qtn1.qw*qtn2.qy - qtn1.qx*qtn2.qy + qtn1.qy*qtn2.qw + qtn1.qz*qtn2.qx
    qtn.qz = qtn1.qw*qtn2.qz + qtn1.qx*qtn2.qy - qtn1.qy*qtn2.qx + qtn1.qz*qtn2.qw
    return qtn

def get_world_coordinate(transform, pose):
    """Get world coordinate by using transform matrix (imu pose)"""
    pose_transform = generate_transform(pose.orientation, pose.position)
    transform = np.dot(pose_transform, transform)
    return transform

def convert_to_world_coordinate(point, transform, stationary_pole):
    """
    Convert to world coordinate by two steps:
    1. from imu to world by using transform matrix (imu pose)
    2. every point substract by the original point to match the visualizer
    """
    trans_point = transform_coordinate(point, transform)
    trans_point.x -= stationary_pole[0]
    trans_point.y -= stationary_pole[1]
    trans_point.z -= stationary_pole[2]
    return trans_point

def euler_to_quaternion(roll, pitch, yaw):
    """Euler to Quaternion"""
    qtnx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qtny = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qtnz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qtnw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qtnx, qtny, qtnz, qtnw]

def quaternion_to_euler(qtnx, qtny, qtnz, qtnw):
    """Quaternion to Euler"""
    tzero = +2.0 * (qtnw * qtnx + qtny * qtnz)
    tone = +1.0 - 2.0 * (qtnx * qtnx + qtny * qtny)
    roll = math.atan2(tzero, tone)
    ttwo = +2.0 * (qtnw * qtny - qtnz * qtnx)
    ttwo = +1.0 if ttwo > +1.0 else ttwo
    ttwo = -1.0 if ttwo < -1.0 else ttwo
    pitch = math.asin(ttwo)
    tthree = +2.0 * (qtnw * qtnz + qtnx * qtny)
    tfour = +1.0 - 2.0 * (qtny * qtny + qtnz * qtnz)
    yaw = math.atan2(tthree, tfour)
    return [yaw, pitch, roll]

def get_interp_pose(timestamp, pose_left, pose_right):
    """Get mean value of two poses"""
    timestamp = float(timestamp)
    gps_pose_left = None if pose_left is None else \
        streaming_utils.load_message_obj(pose_left.objpath)
    gps_pose_right = None if pose_right is None else \
        streaming_utils.load_message_obj(pose_right.objpath)
    if gps_pose_left is None and gps_pose_right is None:
        return None
    elif gps_pose_left is None:
        return GpsSensor(gps_pose_right)
    elif gps_pose_right is None:
        return GpsSensor(gps_pose_left)
    sensor_pose_left = GpsSensor(gps_pose_left)
    sensor_pose_right = GpsSensor(gps_pose_right)
    sensor_pose_interp = GpsSensor(None)
    sensor_pose_interp.position = PointENU()
    interp_in_time = (timestamp - float(pose_left.timestamp)) / \
        (float(pose_right.timestamp) - float(pose_left.timestamp))
    sensor_pose_interp.position.x = sensor_pose_left.position.x + \
        (sensor_pose_right.position.x - sensor_pose_left.position.x) * interp_in_time
    sensor_pose_interp.position.y = sensor_pose_left.position.y + \
        (sensor_pose_right.position.y - sensor_pose_left.position.y) * interp_in_time
    sensor_pose_interp.position.z = sensor_pose_left.position.z + \
        (sensor_pose_right.position.z - sensor_pose_left.position.z) * interp_in_time
    pyqt_left = PyQuaternion(w=sensor_pose_left.orientation.qw,
                             x=sensor_pose_left.orientation.qx,
                             y=sensor_pose_left.orientation.qy,
                             z=sensor_pose_left.orientation.qz)
    pyqt_right = PyQuaternion(w=sensor_pose_right.orientation.qw,
                              x=sensor_pose_right.orientation.qx,
                              y=sensor_pose_right.orientation.qy,
                              z=sensor_pose_right.orientation.qz)
    pyqt_interp = PyQuaternion.slerp(pyqt_left, pyqt_right, amount=interp_in_time)
    sensor_pose_interp.orientation = Quaternion()
    sensor_pose_interp.orientation.qx = pyqt_interp.x
    sensor_pose_interp.orientation.qy = pyqt_interp.y
    sensor_pose_interp.orientation.qz = pyqt_interp.z
    sensor_pose_interp.orientation.qw = pyqt_interp.w
    return sensor_pose_interp

def read_messages_func(record_file):
    """Define a util function to read messages from record file"""
    freader = record.RecordReader(record_file)
    return freader.read_messages()

def get_messages_number(record_file, channels):
    """Return a set of message numbers for specified channels"""
    message_numbers = []
    try:
        freader = record.RecordReader(record_file)
        for channel in channels:
            message_numbers.append(freader.get_messagenumber(channel))
    except Exception:
        return None
    return message_numbers

class Sensor(object):
    """Sensor class representing various of sensors."""
    def __init__(self, channel, intrinsics, extrinsics):
        """Constructor."""
        self._channel = channel
        self._intrinsics = load_yaml_settings(intrinsics)
        self._extrinsics = load_yaml_settings(extrinsics)
        CHANNEL_PROCESS_MAP[self._channel] = self
        if self._extrinsics is None:
            return
        qtn = Quaternion()
        qtn.qw = self._extrinsics['transform']['rotation']['w']
        qtn.qx = self._extrinsics['transform']['rotation']['x']
        qtn.qy = self._extrinsics['transform']['rotation']['y']
        qtn.qz = self._extrinsics['transform']['rotation']['z']
        dev = Point3D()
        dev.x = self._extrinsics['transform']['translation']['x']
        dev.y = self._extrinsics['transform']['translation']['y']
        dev.z = self._extrinsics['transform']['translation']['z']
        self.transform = generate_transform(qtn, dev)

    def process(self, message, timestamp, frame, pose, stationary_pole):
        """Processing function."""
        pass

    def add_transform(self, transform):
        """Add a new transform"""
        self.transform = np.dot(transform, self.transform)

class PointCloudSensor(Sensor):
    """Lidar sensor that hold pointcloud data."""
    def process(self, message, timestamp, frame, pose, stationary_pole):
        """Process PointCloud message."""
        point_cloud = PointCloud()
        point_cloud.ParseFromString(message)
        transform = get_world_coordinate(self.transform, pose)
        for point in point_cloud.point:
            point_world = convert_to_world_coordinate(point, transform, stationary_pole)
            vector4 = frame_pb2.Vector4()
            vector4.x = point_world.x
            vector4.y = point_world.y
            vector4.z = point_world.z
            vector4.i = point.intensity
            point_frame = frame.points.add()
            point_frame.CopyFrom(vector4)
        point = Point3D()
        point.x = 0
        point.y = 0
        point.z = 0
        transform = get_world_coordinate(self.transform, pose)
        point_world = convert_to_world_coordinate(point, transform, stationary_pole)
        frame.device_position.x = point_world.x
        frame.device_position.y = point_world.y
        frame.device_position.z = point_world.z
        rotation = get_rotation_from_tranform(transform)
        qtn = rotation_to_quaternion(rotation)

class RadarSensor(Sensor):
    """Radar sensor that hold radar data."""
    def __init__(self, channel, intrinsics, extrinsics):
        """Initialization."""
        super(RadarSensor, self).__init__(channel, intrinsics, extrinsics)
        self._radar_type = None
        self.transforms = None

    def set_radar_properties(self, radar_type, transforms):
        """Set additional properties for RadarSensor type"""
        self._radar_type = radar_type
        self.transforms = transforms
        for transform in self.transforms:
            self.add_transform(transform)

    def process(self, message, timestamp, frame, pose, stationary_pole):
        """Processing radar message."""
        radar = ContiRadar()
        radar.ParseFromString(message)
        transform = get_world_coordinate(self.transform, pose)
        for point in radar.contiobs:
            point3d = Point3D()
            point3d.x = point.longitude_dist
            point3d.y = point.lateral_dist
            point3d.z = 0
            point_world = convert_to_world_coordinate(point3d, transform, stationary_pole)
            radar_point = frame_pb2.RadarPoint()
            radar_point.type = self._radar_type
            radar_point.position.x = point_world.x
            radar_point.position.y = point_world.y
            radar_point.position.z = point_world.z
            point3d.x = point.longitude_dist + point.longitude_vel
            point3d.y = point.lateral_dist + point.lateral_vel
            point3d.z = 0
            point_world = convert_to_world_coordinate(point3d, transform, stationary_pole)
            radar_point.direction.x = point_world.x - radar_point.position.x
            radar_point.direction.y = point_world.y - radar_point.position.y
            radar_point.direction.z = point_world.z - radar_point.position.z
            radar_frame = frame.radar_points.add()
            radar_frame.CopyFrom(radar_point)

class ImageSensor(Sensor):
    """Image sensor that hold camera data."""
    def __init__(self, channel, intrinsics, extrinsics):
        """Initialization."""
        super(ImageSensor, self).__init__(channel, intrinsics, extrinsics)
        self._task_dir = None
        self.frame_id = 0
        self.transforms = None

    def set_camera_properties(self, task_dir, transforms):
        """Set additional properties for ImageSensor type"""
        self._task_dir = task_dir
        self.transforms = transforms
        for transform in self.transforms:
            self.add_transform(transform)

    def process(self, message, timestamp, frame, pose, stationary_pole):
        """Processing image message."""
        camera_image = frame_pb2.CameraImage()
        camera_image.timestamp = float(timestamp)/(10**9)
        image_name = streaming_utils.get_message_id(timestamp, self._channel)
        dump_img_name(os.path.join(self._task_dir, 'images'), image_name)
        camera_image.image_url = '{}/{}.jpg'.format(SENSOR_PARAMS['image_url'], image_name)
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
        point.x = 0
        point.y = 0
        point.z = 0
        transform = get_world_coordinate(self.transform, pose)
        point_world = convert_to_world_coordinate(point, transform, stationary_pole)
        camera_image.position.x = point_world.x
        camera_image.position.y = point_world.y
        camera_image.position.z = point_world.z
        rotation = get_rotation_from_tranform(transform)
        qtn = rotation_to_quaternion(rotation)
        camera_image.heading.x = qtn.qx
        camera_image.heading.y = qtn.qy
        camera_image.heading.z = qtn.qz
        camera_image.heading.w = qtn.qw
        camera_image.channel = self._channel
        image_frame = frame.images.add()
        image_frame.CopyFrom(camera_image)

class GpsSensor(object):
    """GPS sensor that hold pose data."""
    def __init__(self, message):
        """Initialization."""
        self.position = None
        self.orientation = None
        if message is not None:
            localization = LocalizationEstimate()
            localization.ParseFromString(message)
            self.position = localization.pose.position
            self.orientation = localization.pose.orientation

    def process(self, frame):
        """Process Pose message."""
        gps_pose = frame_pb2.GPSPose()
        gps_pose.lat = self.position.x
        gps_pose.lon = self.position.y
        gps_pose.bearing = self.orientation.qw
        gps_pose.x = self.position.x
        gps_pose.y = self.position.y
        gps_pose.z = self.position.z
        gps_pose.qw = self.orientation.qw
        gps_pose.qx = self.orientation.qx
        gps_pose.qy = self.orientation.qy
        gps_pose.qz = self.orientation.qz
        frame.device_gps_pose.CopyFrom(gps_pose)

class FramePopulator(object):
    """Extract sensors data from record file, and populate to JSON."""
    def __init__(self, root_dir, task_dir, slice_size):
        self._stationary_pole = None
        self._root_dir = root_dir
        self._task_dir = os.path.join(root_dir, task_dir)
        self._slice_size = slice_size
        self._pre_pose_x = None
        self._pre_pose_y = None
        self._pre_frame_time = None
        self._frame_num = 0
        file_utils.makedirs(self._task_dir)

        pointcloud_128 = PointCloudSensor(channel=SENSOR_PARAMS['lidar_channel'],
                                          intrinsics=None,
                                          extrinsics=SENSOR_PARAMS['lidar_extrinsics'])
        image_front_6mm = ImageSensor(channel=SENSOR_PARAMS['front6mm_channel'],
                                      intrinsics=SENSOR_PARAMS['front6mm_intrinsics'],
                                      extrinsics=SENSOR_PARAMS['front6mm_extrinsics'])
        image_front_6mm.set_camera_properties(self._task_dir, [pointcloud_128.transform])

        image_front_12mm = ImageSensor(channel=SENSOR_PARAMS['front12mm_channel'],
                                       intrinsics=SENSOR_PARAMS['front12mm_intrinsics'],
                                       extrinsics=SENSOR_PARAMS['front12mm_extrinsics'])
        image_front_12mm.set_camera_properties(self._task_dir, [pointcloud_128.transform])

        image_left_fisheye = ImageSensor(channel=SENSOR_PARAMS['left_fisheye_channel'],
                                         intrinsics=SENSOR_PARAMS['left_fisheye_intrinsics'],
                                         extrinsics=SENSOR_PARAMS['left_fisheye_extrinsics'])
        image_left_fisheye.set_camera_properties(self._task_dir, [pointcloud_128.transform])

        image_right_fisheye = ImageSensor(channel=SENSOR_PARAMS['right_fisheye_channel'],
                                          intrinsics=SENSOR_PARAMS['right_fisheye_intrinsics'],
                                          extrinsics=SENSOR_PARAMS['right_fisheye_extrinsics'])
        image_right_fisheye.set_camera_properties(self._task_dir, [pointcloud_128.transform])

        image_rear = ImageSensor(channel=SENSOR_PARAMS['rear6mm_channel'],
                                 intrinsics=SENSOR_PARAMS['rear6mm_intrinsics'],
                                 extrinsics=SENSOR_PARAMS['rear6mm_extrinsics'])
        image_rear.set_camera_properties(self._task_dir, [pointcloud_128.transform])

        radar_front = RadarSensor(channel=SENSOR_PARAMS['radar_front_channel'],
                                  intrinsics=None,
                                  extrinsics=SENSOR_PARAMS['radar_front_extrinsics'])
        radar_front.set_radar_properties(frame_pb2.RadarPoint.FRONT, [pointcloud_128.transform])

        radar_rear = RadarSensor(channel=SENSOR_PARAMS['radar_rear_channel'],
                                 intrinsics=None,
                                 extrinsics=SENSOR_PARAMS['radar_rear_extrinsics'])
        radar_rear.set_radar_properties(frame_pb2.RadarPoint.REAR, [pointcloud_128.transform])

    def construct_frames(self, message_structs, max_diff):
        """Construct the frames by using given messages."""
        frame_dir = os.path.join(self._task_dir, 'frames')
        file_utils.makedirs(frame_dir)
        lidar_msg = next(x for x in message_structs
                         if x.message.topic == SENSOR_PARAMS['lidar_channel'])

        lidar_pose = get_interp_pose(lidar_msg.message.timestamp,
                                     lidar_msg.pose_left,
                                     lidar_msg.pose_right)
        frame = frame_pb2.Frame()
        lidar_pose.process(frame)

        if not self.pass_filter_rules(lidar_msg, lidar_pose, message_structs, max_diff):
            return

        if self._stationary_pole is None:
            self._stationary_pole = (lidar_pose.position.x,
                                     lidar_pose.position.y, lidar_pose.position.z)
        file_name = os.path.join(frame_dir, 'frame-{}.json'.format(lidar_msg.message.timestamp))
        if os.path.exists(file_name):
            glog.info('frame file {} already existed, do nothing'.format(file_name))
            return

        for message_struct in message_structs:
            channel, timestamp, _, objpath = message_struct.message
            message_obj = streaming_utils.load_message_obj(objpath)
            pose = get_interp_pose(message_struct.message.timestamp,
                                   message_struct.pose_left,
                                   message_struct.pose_right)
            if isinstance(CHANNEL_PROCESS_MAP[channel], ImageSensor):
                CHANNEL_PROCESS_MAP[channel].frame_id = lidar_msg.message.timestamp
            CHANNEL_PROCESS_MAP[channel].process(message_obj,
                                                 timestamp,
                                                 frame,
                                                 pose,
                                                 self._stationary_pole)
        frame.timestamp = float(lidar_msg.message.timestamp)/(10**9)
        glog.info('converting proto object to json: {}'.format(file_name))
        json_obj = MessageToJson(frame, False, True)
        glog.info('preparing to dump json to file: {}'.format(file_name))
        with open(file_name, 'w') as outfile:
            outfile.write(json_obj)
        glog.info('dumped json: {}'.format(file_name))

    def pass_filter_rules(self, lidar_msg, lidar_pose, message_structs, max_diff):
        """Check if the current frame should be filtered out by various of rules"""
        min_interval = 0.5
        min_distance = 1.0
        max_frame_num = 50
        # Max Diff rule
        if not self.diff_between_lidar_and_camera(lidar_msg, message_structs, max_diff):
            return False
        # Even Interval rule
        if self._pre_frame_time:
            interval = (float(lidar_msg.message.timestamp) - self._pre_frame_time) / (10**9)
            if (interval < min_interval):
                glog.warn('filtered out by interval rule, interval: {}'.format(interval))
                return False
        self._pre_frame_time = float(lidar_msg.message.timestamp)
        # Car Not Moving rule
        if self._pre_pose_x and self._pre_pose_y:
            distance = math.sqrt((float(lidar_pose.position.x) - self._pre_pose_x) ** 2 +
                                 (float(lidar_pose.position.y) - self._pre_pose_y) ** 2)
            if distance < min_distance:
                glog.warn('filtered out by car not moving rule, distance: {}'.format(distance))
                return False
        self._pre_pose_x = float(lidar_pose.position.x)
        self._pre_pose_y = float(lidar_pose.position.y)
        # Max Frames Number rule
        self._frame_num += 1
        if self._frame_num > max_frame_num:
            glog.warn('filtered out by frame number rule, number: {}'.format(self._frame_num))
            return False
        return True

    def diff_between_lidar_and_camera(self, lidar_msg, message_structs, max_diff):
        """Check if time diff between lidar and camera is acceptable"""
        front6mm_msg = next(x for x in message_structs
                            if x.message.topic == SENSOR_PARAMS['front6mm_channel'])
        diff = abs(float(lidar_msg.message.timestamp) - float(front6mm_msg.message.timestamp))
        actual_diff = diff / (10 ** 6)
        if actual_diff > max_diff:
            glog.warn('diff {} is bigger than {}'.format(actual_diff, max_diff))
            return False
        return True

    def format_output(self, file_path, params, tab_size):
        """Output content to file with particular format"""
        with open(file_path, 'a') as output_file:
            line = '{}\n'.format((' ' * tab_size).join([str(param) for param in params]))
            output_file.write(line)

    def construct_bj_frames(self, message_structs, frame_counter, max_diff):
        """Construct frames for labeling pcd/cameras only"""
        channel_map = {
            '/apollo/sensor/camera/front_6mm/image/compressed':'front6mm',
            '/apollo/sensor/camera/front_12mm/image/compressed':'front12mm',
            '/apollo/sensor/camera/left_fisheye/image/compressed':'leftfisheye',
            '/apollo/sensor/camera/right_fisheye/image/compressed':'rightfisheye',
            '/apollo/sensor/camera/rear_6mm/image/compressed':'rear6mm'
        }
        pcd_dir = os.path.join(self._task_dir, 'PCD')
        img_dir = os.path.join(self._task_dir, 'images')
        file_utils.makedirs(pcd_dir)
        file_utils.makedirs(img_dir)
        lidar_msg = next(x for x in message_structs
                         if x.message.topic == SENSOR_PARAMS['lidar_channel'])
        lidar_time = float(lidar_msg.message.timestamp)/(10**9)
        lidar_time_str = '{:.9f}'.format(lidar_time)

        # Filter out the frames that lidar-128 has time diff bigger than designed value
        if not self.diff_between_lidar_and_camera(lidar_msg, message_structs, max_diff):
            glog.warn('keep this frame anyways, and let agent do the filtering per requirement')
            #return

        pcd_file_name = os.path.join(pcd_dir, 'velodyne128-{}.pcd'.format(frame_counter))
        if os.path.exists(pcd_file_name):
            glog.info('frame file {} already existed, do nothing'.format(pcd_file_name))
            return

        # PCD
        glog.info('generating pcd: {}'.format(pcd_file_name))
        pcd_object = streaming_utils.load_message_obj(lidar_msg.message.objpath)
        point_cloud = PointCloud()
        point_cloud.ParseFromString(pcd_object)
        pcd_points = []
        pcd_time = lidar_time
        for point in point_cloud.point:
            pcd_points.append([point.x, point.y, point.z, point.intensity, pcd_time])
        pcd_data = np.array(pcd_points)
        meta_data={}
        meta_data['version'] = '0.7'
        meta_data['fields'] = ['x', 'y', 'z', 'intensity', 'timestamp']
        meta_data['size'] = [4, 4, 4, 1, 4]
        meta_data['type'] = ['F', 'F', 'F', 'U', 'F']
        meta_data['count'] = [1, 1, 1, 1, 1]
        meta_data['width'] = len(pcd_points)
        meta_data['height'] = 1
        meta_data['viewpoint'] = '0 0 0 1 0 0 0'
        meta_data['points'] = len(pcd_points)
        meta_data['data'] = 'ascii'
        pcd = pypcd.PointCloud(meta_data, pcd_data)
        pcd.save_pcd('{}-ascii'.format(pcd_file_name), 'ascii')
        cloud = pypcd.PointCloud.from_path('{}-ascii'.format(pcd_file_name))
        cloud.save_pcd(pcd_file_name, compression='binary_compressed')
        os.remove('{}-ascii'.format(pcd_file_name))

        # IMAGES
        image_file_name = os.path.join(pcd_dir, 'photo.txt')
        glog.info('generating images: {}'.format(image_file_name))
        params = [frame_counter, lidar_time_str]
        for message_struct in message_structs:
            channel, timestamp, _, objpath = message_struct.message
            if channel.find('camera') != -1:
                current_image_files = os.listdir(img_dir)
                results = [e for i, e in enumerate(current_image_files) if e.endswith('.jpg')]
                image_counter = len(results)
                image_data = streaming_utils.load_image_data(objpath)
                img = np.asarray(bytearray(image_data), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                image_name = 'pic-{}.jpg'.format(image_counter)
                cv2.imwrite(os.path.join(img_dir, image_name), img)
                image_name_in_log = '{}_{}'.format(channel_map[channel], image_name)
                if channel == '/apollo/sensor/camera/front_6mm/image/compressed':
                    image_name_in_log = '{:.9f}#{}'.format(float(timestamp) / (10**9), 
                                                           image_name_in_log)
                params.append(image_name_in_log)
        self.format_output(image_file_name, params, 3)

        # POSE
        pose_file_name = os.path.join(pcd_dir, 'pose.txt')
        glog.info('printing gps pose info to: {}'.format(pose_file_name))
        lidar_pose = get_interp_pose(lidar_msg.message.timestamp,
                                     lidar_msg.pose_left,
                                     lidar_msg.pose_right)
        pose_transform = generate_transform(lidar_pose.orientation, lidar_pose.position)
        lidar_transform = CHANNEL_PROCESS_MAP[lidar_msg.message.topic].transform
        transform = np.dot(pose_transform, lidar_transform)
        rotation = get_rotation_from_tranform(transform)
        qtn = rotation_to_quaternion(rotation)
        if not os.path.exists(pose_file_name):
            with open(pose_file_name, 'w') as pose_file:
                line = (' ' * 4).join([str(i) for i in 
                    ['SEQ','TIME','X','Y','Z','QW','QX','QY','QZ']])
                pose_file.write('{}\n'.format(line))
        params = [frame_counter, lidar_time_str, transform[0][3], transform[1][3], transform[2][3],
                  qtn.qw, qtn.qx, qtn.qy, qtn.qz]
        self.format_output(pose_file_name, params, 2)

        #TIMESTAMP
        stamp_file_name = os.path.join(pcd_dir, 'stamp.txt')
        record_file_name = os.path.basename(self._task_dir)
        if not os.path.exists(stamp_file_name):
            with open(stamp_file_name, 'w') as stamp_file:
                line = (' ' * 8).join([str(i) for i in ['RECORD_FILE','SEQ','TIME']])
                stamp_file.write('{}\n'.format(line))
        params = [record_file_name, frame_counter, lidar_time_str]
        self.format_output(stamp_file_name, params, 2)

class DataStream(object):
    """Logic data buffer to manage data reading from different kinds of sources."""
    def __init__(self, data_source, load_func):
        """Initialization. Load the initial data from source"""
        self._data_source = data_source
        self._data_source_index = 0
        self._buffer = []
        self._load_func = load_func
        self._item_number_released = 0
        self._iterators = []
        self.load_more()

    def register(self, iterator):
        """Register iterator so they can be updated correspondingly."""
        self._iterators.append(iterator)

    def read_item(self, index):
        """Read and return a single item by index."""
        if index >= len(self._buffer):
            index -= self.load_more()
        if index < 0 or index >= len(self._buffer):
            return None
        return self._buffer[index]

    def load_more(self):
        """Load more data from source to buffer"""
        if self._data_source_index >= len(self._data_source):
            return 0
        self._buffer.extend(self._load_func(self._data_source[self._data_source_index]))
        self._data_source_index += 1
        # Update each iterator's index if applicable
        item_number_released = len(self._buffer)/3
        for iterator in self._iterators:
            if not iterator.okay_to_update_index(item_number_released+1):
                return 0
        if self._data_source_index <= 1:
            return 0
        del self._buffer[:item_number_released]
        gc.collect()
        for iterator in self._iterators:
            iterator.update_index(item_number_released+1)
        return item_number_released+1

class DataStreamIterator(object):
    """DataStream iterator for accessing the DataStream object."""
    def __init__(self, data_stream):
        """Initialization."""
        self._data_stream = data_stream
        self._index = 0
        self._data_stream.register(self)

    def okay_to_update_index(self, offset):
        """Determine if index is big enough to be updated"""
        return self._index - offset >= 0

    def update_index(self, offset):
        """Update index according to data stream change"""
        self._index -= offset

    def next(self, func):
        """Get next item that satisfy func."""
        item = self._data_stream.read_item(self._index)
        while item is not None and not func(item):
            self._index += 1
            item = self._data_stream.read_item(self._index)
        self._index += 1
        return item

class MessageStruct(object):
    """Data structure representing messages with left and right poses."""
    def __init__(self, msg, pose_left, pose_right):
        self.message = msg
        self.pose_left = pose_left
        self.pose_right = pose_right

class Builder(object):
    """Used for building objects with specific sequences and properties."""
    def __init__(self, message_struct, rules):
        self._guide_lines = {}
        for rule in rules:
            self._guide_lines[rule] = None
        self.build(message_struct)

    def build(self, message_struct):
        """Check if the message can be accepted. Return 1 if yes, 0 if not."""
        topic = message_struct.message.topic
        if self._guide_lines[topic] is None:
            self._guide_lines[topic] = message_struct
            if all(self._guide_lines[x] is not None for x in self._guide_lines):
                return 1, None  # means message accepted and builder done
            return 0, None   # means message accepted
        return 0, message_struct # means message not accepted

    def complete(self, frame_populator, frame_counter, agent, diff):
        """Builder complete, and send messages to framepopulator in this case"""
        messages = self._guide_lines.values()
        messages = sorted(messages, key=lambda message_struct: message_struct.message.topic)
        if agent == "scale":
            frame_populator.construct_frames(messages, diff)
        elif agent == "bj":
            frame_populator.construct_bj_frames(messages, frame_counter, diff)

class BuilderManager(object):
    """Builder management pool."""
    def __init__(self, rules, frame_populator):
        self._builder_list = []
        self._rules = rules
        self._frame_populator = frame_populator
        self._counter = 1

    def throw_to_pool(self, message_struct, agent, diff):
        """Process new coming message. Loop each builder in the list and find the right one"""
        for builder in self._builder_list:
            status, msg = builder.build(message_struct)
            if msg is None:
                if status == 1:
                    glog.info('constructing the {}th frame'.format(self._counter))
                    builder.complete(self._frame_populator, self._counter, agent, diff)
                    self._counter += 1
                    self._builder_list.remove(builder)
                return
        self._builder_list.append(Builder(message_struct, self._rules))
