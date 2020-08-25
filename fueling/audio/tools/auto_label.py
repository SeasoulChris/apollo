#!/usr/bin/env python
"""
Please read the global configuration variables carefully, then
Run with:
    bazel run //fueling/audio/tools:auto_label -- --cloud
"""

# Python packages
from collections import namedtuple
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import math
import os
import yaml

# Apollo-fuel packages
from fueling.audio.tools.direction_detection import get_direction
from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.logging as logging

# To extract direction detection data directly from audio channel, set this to False
# Otherwise, will estimate from python code -- ./direction_detection.py
DIRECTION_FROM_RAW_DATA = True
TARGET_OBSTACLE_ID = [1165, 1174]
RECORD_PREFIX = 'public-test/2020/2020-08-24/2020-08-24-14-58-13'
EXTRINSIC_FILE_PATH = 'modules/audio/conf/respeaker_extrinsics.yaml'
OUTPUT_DIR = 'modules/audio/metrics'

Qtn = namedtuple('Quaternion', ['qx', 'qy', 'qz', 'qw'])
Point3D = namedtuple('Point3D', ['x', 'y', 'z'])


def generate_transform(qtn, dev):
    """Generate a matrix with rotation and deviation/translation."""
    transform = np.zeros(shape=(4, 4), dtype=float)
    transform[0][0] = qtn.qw**2 + qtn.qx**2 - qtn.qy**2 - qtn.qz**2
    transform[0][1] = 2 * (qtn.qx * qtn.qy - qtn.qw * qtn.qz)
    transform[0][2] = 2 * (qtn.qx * qtn.qz + qtn.qw * qtn.qy)
    transform[1][0] = 2 * (qtn.qx * qtn.qy + qtn.qw * qtn.qz)
    transform[1][1] = qtn.qw**2 - qtn.qx**2 + qtn.qy**2 - qtn.qz**2
    transform[1][2] = 2 * (qtn.qy * qtn.qz - qtn.qw * qtn.qx)
    transform[2][0] = 2 * (qtn.qx * qtn.qz - qtn.qw * qtn.qy)
    transform[2][1] = 2 * (qtn.qy * qtn.qz + qtn.qw * qtn.qx)
    transform[2][2] = qtn.qw**2 - qtn.qx**2 - qtn.qy**2 + qtn.qz**2
    transform[0][3] = dev.x
    transform[1][3] = dev.y
    transform[2][3] = dev.z
    transform[3] = [0, 0, 0, 1]
    return transform


class DirectionAutoLabeler(BasePipeline):

    def run(self):
        self.to_rdd(self.our_storage().list_files(RECORD_PREFIX)).filter(
            record_utils.is_record_file).foreach(self.calculate)

    def calculate(self, record):
        """
        Calculate accuracy between predited direction & actual one
        """
        logging.info("Processing record: {}".format(record))
        reader = record_utils.read_record(
            [record_utils.OBSTACLES_CHANNEL, record_utils.MICROPHONE_CHANNEL,
                record_utils.LOCALIZATION_CHANNEL, record_utils.AUDIO_CHANNEL])
        # EV: [[timestampes, ...], [points, ...]]
        ev_true = [[], []]
        ev_preds = [[], []]
        # Car: [[timestampes, ...], [locations, ...], [orientations, ...]]
        car_pose = [[], [], []]
        for msg in reader(record):
            if msg.topic == record_utils.OBSTACLES_CHANNEL:
                obstacles = record_utils.message_to_proto(msg)
                timestamp = obstacles.header.timestamp_sec
                obstacles = [
                    obs for obs in obstacles.perception_obstacle if obs.id in TARGET_OBSTACLE_ID]
                if obstacles:
                    ev_true[0] += timestamp,
                    ev_true[1] += obstacles[0].position,
            elif msg.topic == record_utils.MICROPHONE_CHANNEL and DIRECTION_FROM_RAW_DATA:
                microphone_data = record_utils.message_to_proto(msg)
                ev_preds[0] += microphone_data.header.timestamp_sec,
                ev_preds[1] += get_direction(microphone_data),
            elif msg.topic == record_utils.AUDIO_CHANNEL and not DIRECTION_FROM_RAW_DATA:
                audio_detection = record_utils.message_to_proto(msg)
                ev_preds[0] += audio_detection.header.timestamp_sec,
                ev_preds[1] += audio_detection.position,
            elif msg.topic == record_utils.LOCALIZATION_CHANNEL:
                loc_data = record_utils.message_to_proto(msg)
                car_pose[0] += loc_data.header.timestamp_sec,
                car_pose[1] += loc_data.pose.position,
                car_pose[2] += loc_data.pose.orientation,
        yaml_file_path = self.our_storage().abs_path(EXTRINSIC_FILE_PATH)
        self.respeaker_extrinsic = yaml.safe_load(open(yaml_file_path))

        # If data is estimated from raw microphone data rather than audio module
        if DIRECTION_FROM_RAW_DATA:
            self.degree2point(ev_preds)

        base_ = os.path.basename(record)
        base_ = "{}.metric".format(base_)
        self.evaluate(ev_true, ev_preds, car_pose, base_)

    def evaluate(self, ev_true, ev_preds, car_pose, output_file_name):
        """
        Evaluate between predicting degree (ev_pred) & true degree (ev_true)
        """
        if len(ev_true[0]) <= 1 or len(ev_preds[0]) <= 1 or len(car_pose[0]) <= 1:
            logging.info("No Enough data for aligning, exiting...")
            return

        # Align all data
        ev_true, ev_preds, car_pose = self.interpolate(
            ev_preds, ev_true, car_pose)
        # Obstacle's coordinate is world-based, so convert to respeaker's coordinate
        ev_true[1] = self.world2respeaker(ev_true, car_pose)

        # Convert those points to degree for comparison
        ev_true[1] = self.point2degree(ev_true[1])
        ev_preds[1] = self.point2degree(ev_preds[1])

        output_dir = self.our_storage().abs_path(OUTPUT_DIR)
        output_path = os.path.join(output_dir, output_file_name)
        logging.info("Writing to {}".format(output_path))
        self.write_to_file(output_path, ev_preds, ev_true)

    def interpolate(self, ev_preds, ev_true, car_pose, freq=5):
        """Align data to a certain frequency"""
        def point3d2matrix(points):
            """From Point3d(x, y, z) to [[x], [y], [z], [1]]"""
            for i, p in enumerate(points):
                points[i] = [[p.x], [p.y], [p.z], [1]]

        point3d2matrix(ev_preds[1])
        point3d2matrix(ev_true[1])
        point3d2matrix(car_pose[1])

        start = max(ev_true[0][0], ev_preds[0][0], car_pose[0][0])
        end = min(ev_true[0][-1], ev_preds[0][-1], car_pose[0][-1])
        n = int(end - start) * freq

        ev_true = self.interpolate_points(*ev_true, [start, end], n)
        ev_preds = self.interpolate_points(*ev_preds, [start, end], n)
        car_pose = self.interpolate_orientation(*car_pose, [start, end], n)

        return ev_true, ev_preds, car_pose

    def interpolate_points(self, timestamps, points, time_interval, n):
        """Interpolate n 3d points within time_interval"""
        f = interp1d(timestamps, points, axis=0)
        new_timestamps = np.linspace(*time_interval, n)

        return [new_timestamps, f(timestamps)]

    def interpolate_orientation(self, timestamps, locs, qtns, time_interval, n):
        new_timestamps, locs = self.interpolate_points(
            timestamps, locs, time_interval, n)
        qtns = R.from_quat([[qtn.qx, qtn.qy, qtn.qz, qtn.qw] for qtn in qtns])
        f = Slerp(timestamps, qtns)

        return [new_timestamps, locs, f(new_timestamps).as_quat()]

    def world2respeaker(self, ev, car_pose):
        """From world coordinate to respeaker coordinate"""
        imu2respeaker = self.get_imu2respeaker_transformation()
        ev_positions = []
        for t, ev_position, _, car_position, car_orientation in zip(*ev, *car_pose):
            # check if this is right...
            imu2world = generate_transform(
                Qtn(*car_orientation), Point3D(*car_position.flatten()[:3]))
            world2imu = np.linalg.inv(imu2world)
            ev_positions += np.dot(imu2respeaker,
                                   np.dot(world2imu, ev_position)),

        return ev_positions

    def degree2point(self, EV):
        """Assuming distance is 50 m"""
        for i, degree in enumerate(EV[1]):
            degree /= math.pi
            EV[1][i] = Point3D(50 * math.sin(degree), 50 * math.cos(degree), 0)

    def point2degree(self, points):
        """Accept points like [[x], [y], [z]]"""
        ev_degrees = []
        for p in points:
            x, y = p[0][0], p[1][0]
            degree = math.asin(y / (x * x + y * y) ** 0.5) / math.pi * 90
            if x < 0:
                degree = 180 - degree
            ev_degrees += degree % 360,
        return ev_degrees

    def get_imu2respeaker_transformation(self):
        qtn = self.respeaker_extrinsic["transform"]["rotation"]
        dev = self.respeaker_extrinsic["transform"]["translation"]
        transform = generate_transform(
            Qtn(qtn["x"], qtn["y"], qtn["z"], qtn["w"]), Point3D(dev["x"], dev["y"], dev["z"]))
        inv_transform = np.linalg.inv(transform)

        return inv_transform

    def write_to_file(self, file_path, ev_preds, ev_true):
        points = "Timestamp\tPrediction\tTrue Label\n"
        deviations = []
        for t1, d1, d2 in zip(ev_preds[0], ev_preds[1], ev_true[1]):
            points += "{}\t{}\t{}\n".format(t1, d1, d2)
            deviations.append(abs(d1 - d2) % 360)
        stat = "Avg deviation: {}\t, Standard deviation: {}\n".format(
            np.average(deviations), np.std(deviations))
        with open(file_path, "w") as f:
            f.write(stat + points)


def local_test_file():
    DirectionAutoLabeler().calculate(
        "/apollo/public_test/2020-08-24-14-58-13/20200824145813.record.00005")


def local_test_dir():
    autolabeler = DirectionAutoLabeler()
    for root, dirs, files in os.walk("/apollo/public_test/2020-08-24-14-58-13"):
        for file in files:
            path = os.path.join(root, file)
            logging.info("Processing {}".format(path))
            autolabeler.calculate(path)


if __name__ == '__main__':
    DirectionAutoLabeler().main()
    # local_test_file()
    # local_test_dir()
