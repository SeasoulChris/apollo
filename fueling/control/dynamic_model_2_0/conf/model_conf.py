#!/usr/bin/env python

segment_index = {
    "heading": 0,  # pose.heading, ENU
    "q_x": 1,  # pose.orientation.qx, ENU
    "q_y": 2,  # pose.orientation.qy, ENU
    "q_z": 3,  # pose.orientation.qz, ENU
    "q_w": 4,  # pose.orientation.qw, ENU
    "v_x": 5,  # pose.linear_velocity.x, ENU
    "v_y": 6,  # pose.linear_velocity.y, ENU
    "v_z": 7,  # pose.linear_velocity.z, ENU
    "a_x": 8,  # pose.linear_acceleration.x, ENU
    "a_y": 9,  # pose.linear_acceleration.y, ENU
    "a_z": 10,  # pose.linear_acceleration.z, ENU
    "w_x": 11,  # pose.angular_velocity.x, ENU
    "w_y": 12,  # pose.angular_velocity.y, ENU
    "w_z": 13,  # pose.angular_velocity.z, ENU
    "speed": 14,  # chassis.speed_mps
    "throttle": 15,  # chassis.throttle_percentage/100.0
    "brake": 16,  # chassis.brake_percentage/100.0
    "steering": 17,  # chassis.steering_percentage/100.0
    "mode": 18,  # chassis.driving_mode
    "x": 19,  # pose.position.x, ENU, ground truth from localization module
    "y": 20,  # pose.position.y, ENU, ground truth from localization module
    "z": 21,  # pose.position.z, ENU, ground truth from localization module
    "gear_position": 22  # gear position: 0-neutral, 1-drive, 2-reverse
}

feature_config = {
    "input_dim": 6,  # Input dimension of DM 2.0
    "output_dim": 2,  # Output dimension
    "mlp_input_dim": 5,  # Input dimension of DM 1.0
    "delta_t": 0.01,  # updating cycle delta_t for input data
    "DELTA_T": 1.0,  # updating cycle DELTA_T for output data (residual correction)
    "window_size": 51,  # window_size for savgol_filter
    "polynomial_order": 3,  # polynomial_order for savgol_filter
    "SEGMENT_OVERLAP": 10,  # The overlap between each two groups for augment
    "MAX_POSE_DELTA": 0.02,  # The maximum gap between pose and chasis timestamp, in seconds
    "COMPENSATE_CHASIS": False,  # Whether compensate chasis with poses
    "CHASIS_DELTA_T": 0.011,  # The maximum gap between chasis messages, in seconds
}

""" Input index is a 2-D matrix of size [sequence_length][input_feature_dim]"""
""" sequence_length = DELTA_T / delta_t = 100, input_feature_dim = 5"""
input_index = {
    "v": 0,  # chassis.speed_mps
    "a": 1,  # a_x * cos(heading) + a_y * sin(heading)
    "u_1": 2,  # chassis.throttle_percentage/100.0
    "u_2": 3,  # chassis.brake_percentage/100.0
    "u_3": 4,  # chassis.steering_percentage/100.
    "phi": 5  # pose.heading, ENU
}

output_index = {
    "d_x": 0,  # the residual error between groud-truth x and predicted x
    "d_y": 1  # the residual error between groud-truth y and predicted y
}

imu_scaling = {
    # IMU compensation for localization scaling issue
    "pp6": 0.55,  # scaling rate for acceleration and angular speed on pp6
    "pp7": 1.085,  # scaling rate for acceleration and angular speed on pp7
    "acc": 0.76,  # scaling rate for acceleration
    "heading_rate": 0.78  # scaling rate for angular speed
}
