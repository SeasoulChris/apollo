#!/usr/bin/env python

""" Open-space planner feature extraction related utils. """


import numpy as np
import math

from shapely.geometry import Polygon, LineString, Point

import fueling.common.logging as logging
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils
from fueling.profiling.conf.open_space_planner_conf import FEATURE_IDX, REFERENCE_VALUES


def delta(feature_name, prev_feature, curr_feature):
    feature_idx = FEATURE_IDX[feature_name]
    return curr_feature[feature_idx] - prev_feature[feature_idx]


def calc_lon_acc_bound(acc_lat, dec_lat):
    if acc_lat == 0.0:
        return 0.5 * (dec_lat + 3.0)
    return -0.5 * (acc_lat - 3.0)


def calc_lon_dec_bound(acc_lat, dec_lat):
    if acc_lat == 0.0:
        return -2.0 / 3.0 * (dec_lat + 3.0)
    return 2.0 / 3.0 * (acc_lat - 3.0)


def calc_lat_acc_bound(acc_lon, dec_lon):
    if acc_lon == 0.0:
        return 3.0 / 2.0 * dec_lon + 3.0
    return -2.0 * acc_lon + 3.0


def calc_lat_dec_bound(acc_lon, dec_lon):
    if acc_lon == 0.0:
        return -3.0 / 2.0 * dec_lon - 3.0
    return 2 * acc_lon - 3.0


def steer_limit(steer_val, vehicle_param):
    return steer_val / vehicle_param.steer_ratio / vehicle_param.wheel_base


def transform(polygon, position, heading, new_position, new_heading):
    """
    2D Rigid body tranformation matrix T = [[cos -sin x],
                                            [sin  cos y],
                                            [ 0    0  1]]
    Point(p2 w.r.t world) = T(p2 w.r.t world)*T(world w.r.t p1)*Point(p1 w.r.t world)
    """
    cur_T = ([math.cos(heading), -math.sin(heading), position[0]],
             [math.sin(heading), math.cos(heading), position[1]],
             [0, 0, 1])
    new_T = ([math.cos(new_heading), -math.sin(new_heading), new_position[0]],
             [math.sin(new_heading), math.cos(new_heading), new_position[1]],
             [0, 0, 1])
    return np.dot(polygon, np.dot(new_T, np.linalg.inv(cur_T)).T)


def compute_adc_points(position, heading, vehicle_param, z=1):
    front_edge_to_center = vehicle_param.front_edge_to_center
    back_edge_to_center = vehicle_param.back_edge_to_center
    length = vehicle_param.length
    width = vehicle_param.width

    offset = 0.0
    if front_edge_to_center > back_edge_to_center:
        offset = front_edge_to_center - length / 2.0
    else:
        offset = back_edge_to_center - length / 2.0
    center_x = position[0] + offset * math.cos(heading)
    center_y = position[1] + offset * math.sin(heading)

    dx1 = math.cos(heading) * length / 2.0
    dy1 = math.sin(heading) * length / 2.0
    dx2 = math.sin(heading) * width / 2.0
    dy2 = -math.cos(heading) * width / 2.0
    return [[center_x + dx1 + dx2, center_y + dy1 + dy2, z],
            [center_x + dx1 - dx2, center_y + dy1 - dy2, z],
            [center_x - dx1 - dx2, center_y - dy1 - dy2, z],
            [center_x - dx1 + dx2, center_y - dy1 + dy2, z]]


def extract_obstacle_polygon_points(perception_obstacle, z=1):
    """
    Returns a list of polygon points representing one obstacle
    """
    points = getattr(perception_obstacle, 'polygon_point', [])
    return [[point.x, point.y, z] for point in points]


def find_min_collision_time(msg, vehicle_param):
    """
    Returns the minimum time to collision for a planning trajectory in relatvie time
    """
    min_collision_time = REFERENCE_VALUES['max_time_to_collision']
    prediction = msg['prediction']
    obstacle_start_time = prediction.header.timestamp_sec
    planning = msg['planning']
    adc_start_time = planning.header.timestamp_sec

    for traj_point in planning.trajectory_point:
        # exclude past or stationary trajectory points
        if (traj_point.relative_time < 0) or (abs(traj_point.v) < 0.1):
            continue
        adc_time = adc_start_time + traj_point.relative_time
        path_point = traj_point.path_point
        adc_polygon = compute_adc_points(
            [path_point.x, path_point.y], path_point.theta, vehicle_param)

        for obstacle in prediction.prediction_obstacle:
            cur_polygon = extract_obstacle_polygon_points(obstacle.perception_obstacle)
            cur_position = obstacle.perception_obstacle.position
            if Polygon(adc_polygon).intersects(Polygon(cur_polygon)):
                logging.info(F'{adc_start_time}s: Found collision at {traj_point.relative_time} '
                             F'with {obstacle.perception_obstacle.id}')
                min_collision_time = min(traj_point.relative_time, min_collision_time)
                continue

            if not obstacle.trajectory:
                continue

            # TODO sort prediction trajectory based on probability
            predicted_traj = obstacle.trajectory[0]
            for predicted_pt in predicted_traj.trajectory_point:
                predicted_time = obstacle_start_time + predicted_pt.relative_time
                if abs(adc_time - predicted_time) < 0.1:
                    predicted_polygon = transform(cur_polygon,
                                                  [cur_position.x,
                                                      cur_position.y], obstacle.perception_obstacle.theta,
                                                  [predicted_pt.path_point.x, predicted_pt.path_point.y],
                                                  predicted_pt.path_point.theta)
                    if Polygon(adc_polygon).intersects(Polygon(predicted_polygon)):
                        logging.info(F'{adc_start_time}s: Predicted collision at '
                                     F'{traj_point.relative_time} '
                                     F'with {obstacle.perception_obstacle.id}')
                        min_collision_time = min(traj_point.relative_time, min_collision_time)
                        # ignore further predicted points
                        break

        # collision occurs at traj_point, ignore further trajectory points
        if min_collision_time < REFERENCE_VALUES['max_time_to_collision']:
            break
    return min_collision_time


"""
 trajectory point example:
 path_point:
     x: 559787.066030095
     y: 4157751.813925536
     z: 0.000000000
     theta: 2.379002832
     kappa: -0.019549716
     s: -2.356402468
     dkappa: 0.000000000
     ddkappa: 0.000000000
 v: 4.474370468
 a: 0.995744297
 relative_time: -0.400000000
"""


def extract_data_from_trajectory_point(trajectory_point, vehicle_param, roi_boundaries, obstacles,
                                       min_collision_time):
    """Extract fields from a single trajectory point"""
    path_point = trajectory_point.path_point
    speed = trajectory_point.v
    a = trajectory_point.a
    # get lateral acc and dec
    lat = speed * speed * path_point.kappa
    if lat > 0.0:
        lat_acc = lat
        lat_dec = 0.0
    else:
        lat_acc = 0.0
        lat_dec = lat
    # get longitudinal acc and dec
    if a > 0.0:
        lon_acc = math.sqrt(abs(a**2 - (lat)**2))
        lon_dec = 0.0
    else:
        lon_acc = 0.0
        lon_dec = -1.0 * math.sqrt(abs(a**2 - (lat)**2))

    # calculate comfort bound
    lon_acc_bound = calc_lon_acc_bound(lat_acc, lat_dec)
    lon_dec_bound = calc_lon_dec_bound(lat_acc, lat_dec)
    lat_acc_bound = calc_lat_acc_bound(lon_acc, lon_dec)
    lat_dec_bound = calc_lat_dec_bound(lon_acc, lon_dec)

    # distance
    if trajectory_point.relative_time < 0:
        distance_to_roi_ratio = 0
        distance_to_obstacle_ratio = 0
    else:
        traj_point = Point(path_point.x, path_point.y)
        distances_to_roi = [boundary.distance(traj_point) for boundary in roi_boundaries]
        distance_to_roi_ratio = REFERENCE_VALUES['roi_reference_distance'] / \
            min(distances_to_roi) if distances_to_roi else 0.0

        distance_to_obstacles = [obstacle.distance(traj_point) for obstacle in obstacles]
        distance_to_obstacle_ratio = REFERENCE_VALUES['obstacle_reference_distance'] / min(
            distance_to_obstacles) if distance_to_obstacles else 0.0

    if min_collision_time == REFERENCE_VALUES['max_time_to_collision']:
        time_to_collision = min_collision_time  # no collision
    elif trajectory_point.relative_time > min_collision_time:
        time_to_collision = 0  # collision has already occurred
    else:
        time_to_collision = min_collision_time - trajectory_point.relative_time

    if hasattr(trajectory_point, 'relative_time'):
        # NOTE: make sure to update TRAJECTORY_FEATURE_NAMES in
        # open_space_planner_conf.py if updating this data array.
        # Will need a better way to sync these two pieces.
        data_array = np.array([
            trajectory_point.relative_time,
            path_point.kappa,
            abs(path_point.kappa) / steer_limit(vehicle_param.max_steer_angle, vehicle_param),
            speed,  # not sure if needed
            a,
            lon_acc if a > 0.0 else lon_dec,
            lat_acc if lat > 0.0 else lat_dec,

            # ratios
            a / vehicle_param.max_acceleration if a > 0.0 else 0.0,
            a / vehicle_param.max_deceleration if a < 0.0 else 0.0,
            lon_acc / lon_acc_bound,
            lon_dec / lon_dec_bound,
            lat_acc / lat_acc_bound,
            lat_dec / lat_dec_bound,
            distance_to_roi_ratio,
            distance_to_obstacle_ratio,
            time_to_collision / REFERENCE_VALUES['time_to_collision'],
        ])
    return data_array


def calculate_jerk_ratios(prev_feature, curr_feature):
    if prev_feature is None or curr_feature is None:
        return [0.0, 0.0, 0.0, 0.0]

    delta_t = delta('relative_time', prev_feature, curr_feature)
    lon_jerk = delta('longitudinal_acceleration', prev_feature, curr_feature) / delta_t
    if lon_jerk > 0.0:
        lon_pos_jerk = lon_jerk
        lon_neg_jerk = 0.0
    else:
        lon_pos_jerk = 0.0
        lon_neg_jerk = lon_jerk

    lat_jerk = delta('lateral_acceleration', prev_feature, curr_feature) / delta_t
    if lat_jerk > 0.0:
        lat_pos_jerk = lat_jerk
        lat_neg_jerk = 0.0
    else:
        lat_pos_jerk = 0.0
        lat_neg_jerk = lat_jerk

    return [
        lon_pos_jerk / REFERENCE_VALUES['longitudinal_jerk_positive_upper_bound'],
        lon_neg_jerk / REFERENCE_VALUES['longitudinal_jerk_negative_upper_bound'],
        lat_pos_jerk / REFERENCE_VALUES['lateral_jerk_positive_upper_bound'],
        lat_neg_jerk / REFERENCE_VALUES['lateral_jerk_negative_upper_bound'],
    ]


def calculate_dkappa_ratio(prev_feature, curr_feature, vehicle_param):
    if prev_feature is None or curr_feature is None:
        return 0.0

    delta_t = delta('relative_time', prev_feature, curr_feature)
    dkappa = delta('kappa', prev_feature, curr_feature) / delta_t
    dkappa_ratio = abs(dkappa) / steer_limit(vehicle_param.max_steer_angle_rate, vehicle_param),
    return [dkappa_ratio]


def extract_data_from_trajectory(trajectory, vehicle_param, roi_boundaries, obstacles,
                                 min_collision_time):
    """Extract data from all trajectory points"""
    feature_list = []
    prev_features = None
    for trajectory_point in trajectory:
        features = extract_data_from_trajectory_point(
            trajectory_point, vehicle_param, roi_boundaries, obstacles, min_collision_time)
        if features is None:
            continue

        features = np.append(features, calculate_jerk_ratios(prev_features, features))
        features = np.append(
            features,
            calculate_dkappa_ratio(
                prev_features,
                features,
                vehicle_param))
        feature_list.append(features)
        prev_features = features

    trajectory_mtx = np.array(feature_list)
    return trajectory_mtx


def extract_roi_boundaries(msgs):
    boundaries = []
    for msg in msgs:
        open_space_msg = msg['planning'].debug.planning_data.open_space
        if open_space_msg.obstacles:
            origin = open_space_msg.origin_point

            # Not a mistake, ROI boundaries are stored as open_space obstacles
            for roi_pb2 in open_space_msg.obstacles:
                boundary = [(x + origin.x, y + origin.y)
                            for (x, y) in zip(roi_pb2.vertices_x_coords, roi_pb2.vertices_y_coords)
                            ]
                boundaries.append(LineString(boundary))
            break

    return boundaries


def extract_obstacle_polygons(msg):
    """
    Returns a list of shapely.geometry.Polygon objects that represents all the obstacles
    """
    if 'prediction' not in msg:
        return []

    def get_polygon(obstacle):
        points = getattr(obstacle.perception_obstacle, 'polygon_point', [])
        return Polygon([(point.x, point.y) for point in points])

    obstacles = getattr(msg['prediction'], 'prediction_obstacle', [])
    return [get_polygon(obstacle) for obstacle in obstacles]


def extract_planning_trajectory_feature(target_groups):
    """Extract planning trajectory related feature matrix from a group of planning messages"""
    target, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')

    vehicle_param = multi_vehicle_utils.get_vehicle_param(target)
    roi_boundaries = extract_roi_boundaries(msgs)

    extracted_data = (extract_data_from_trajectory(
        msg['planning'].trajectory_point, vehicle_param, roi_boundaries, extract_obstacle_polygons(
            msg),
        find_min_collision_time(msg, vehicle_param))
        for msg in msgs)
    planning_trajectory_mtx = np.concatenate(
        [data for data in extracted_data if data is not None and data.shape[0] > 10])

    return target, planning_trajectory_mtx


def extract_meta_from_planning(msg):
    """Extract non-repeated field from one planning message"""
    zigzag_latency = 0.0
    if msg.debug.planning_data.open_space.time_latency:
        zigzag_latency = msg.debug.planning_data.open_space.time_latency
    meta_array = np.array([
        msg.latency_stats.total_time_ms,  # end-to-end time latency
        zigzag_latency,  # zigzag trajectory time latency
    ])
    return meta_array


def extract_latency_feature(target_groups):
    """Extract latency related feature matrix from a group of planning messages"""
    target, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')
    latency_mtx = np.array([data for data in [extract_meta_from_planning(msg['planning'])
                                              for msg in msgs] if data is not None])
    return target, latency_mtx


def compute_path_length(trajectory):
    trajectory_points = trajectory.trajectory_point
    if len(trajectory_points) < 2:
        return -1
    return abs(trajectory_points[-1].path_point.s - trajectory_points[0].path_point.s)


def extract_data_from_zigzag(msg, wheel_base):
    """Extract open space debug from one planning message"""
    data = []
    for zigzag in msg.debug.planning_data.open_space.partitioned_trajectories.trajectory:
        path_length = compute_path_length(zigzag)
        if path_length == -1 or path_length == 0.0:
            data.append(-1)  # no zigzag path length
        else:
            data.append(wheel_base / path_length)

    return data


def extract_zigzag_trajectory_feature(target_groups):
    """Extract zigzag trajectory related feature matrix from a group of planning messages"""
    target, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')

    vehicle_param = multi_vehicle_utils.get_vehicle_param(target)

    zigzag_list = []
    for msg in msgs:
        # Find the first frame when zigzag trajectory is ready
        # Do not depend on the number of zigzag trajectories as it may be a stopping one
        if msg['planning'].debug.planning_data.open_space.time_latency > 0:
            zigzag_list.extend(extract_data_from_zigzag(msg['planning'], vehicle_param.wheel_base))
    return target, np.array([zigzag_list]).T  # make sure numpy shape is (num, 1)


def extract_stage_feature(target_groups):
    """Extract scenario stage related feature matrix from a group of planning messages"""
    target, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')

    start_timestamp = msgs[0]['planning'].header.timestamp_sec
    end_timestamp = msgs[-1]['planning'].header.timestamp_sec
    gear_shift_times = 1
    for msg in msgs:
        # Find the first frame when zigzag trajectory is ready
        # Do not depend on the number of zigzag trajectories as it may be a stopping one
        if msg['planning'].debug.planning_data.open_space.time_latency > 0:
            gear_shift_times = len(
                msg['planning'].debug.planning_data.open_space.partitioned_trajectories.trajectory)
            for point in msg['planning'].trajectory_point:
                if point.relative_time == 0.0:
                    initial_heading = point.path_point.theta
                    break
            break
    stage_completion_time = (end_timestamp - start_timestamp) / gear_shift_times * 1000.0

    actual_heading = msgs[0]['planning'].debug.planning_data.adc_position.pose.heading
    vehicle_param = multi_vehicle_utils.get_vehicle_param(target)
    initial_heading_diff_ratio = abs(initial_heading - actual_heading) \
        / (vehicle_param.max_steer_angle / vehicle_param.steer_ratio)
    return target, np.array([[stage_completion_time, initial_heading_diff_ratio]])
