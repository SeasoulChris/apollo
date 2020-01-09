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
    "x": 19,  # pose.position.x, ENU
    "y": 20,  # pose.position.y, ENU
    "z": 21,  # pose.position.z, ENU
    "gear_position": 22  # gear position: 0-neutral, 1-drive, 2-reverse
}

input_index = {
    "speed": 0,  # chassis.speed_mps
    "acceleration": 1,  # a_x * cos(heading) + a_y * sin(heading)
    "throttle": 2,  # chassis.throttle_percentage/100.0
    "brake": 3,  # chassis.brake_percentage/100.0
    "steering": 4  # chassis.steering_percentage/100.
}

# holistic dynamic model by taking lateral speed and acceleration in account
holistic_input_index = {
    "lon_speed": 0,  # v_x * cos(heading) + v_y * sin(heading), longitudinal speed in RFU
    "lat_speed": 1,  # v_x * sin(heading) - v_y * cos(heading), lateral speed in RFU
    # a_x * cos(heading) + a_y * sin(heading), longitudinal acceleration in RFU
    "lon_acceleration": 2,
    "lat_acceleration": 3,  # a_x * sin(heading) - a_y * cos(heading), lateral acceleration in RFU
    "w_z": 4,  # a_x * sin(heading) - a_y * cos(heading), lateral acceleration in RFU
    "throttle": 5,  # chassis.throttle_percentage/100.0
    "brake": 6,  # chassis.brake_percentage/100.0
    "steering": 7  # chassis.steering_percentage/100.
}

output_index = {
    "acceleration": 0,  # a_x * cos(heading) + a_y * sin(heading)
    "w_z": 1  # pose.angular_velocity.z
}


pose_output_index = {
    "acceleration": 0,  # a_x * cos(heading) + a_y * sin(heading)
    "w_z": 1,  # pose.angular_velocity.z
    "speed": 2
}

# holistic dynamic model by taking lateral speed and acceleration in account
holistic_output_index = {
    # a_x * cos(heading) + a_y * sin(heading), longitudinal acceleration in RFU
    "lon_acceleration": 0,
    "lat_acceleration": 1,  # a_x * sin(heading) - a_y * cos(heading), lateral acceleration in RFU
    "w_z": 2  # pose.angular_velocity.z
}

feature_config = {
    "is_holistic": 0,  # if the current dynamic model is holistic
    "is_backward": 1,  # if the feature is for backward driving scenario
    "vehicle_id": 'ch1',  # the vehicle id for the feature
    "input_dim": 5,  # input feature dimension
    "holistic_input_dim": 8,  # holistic input feature dimension
    "output_dim": 2,  # output feature dimension
    "holistic_output_dim": 3,  # holistic output feature dimension
    "maximum_segment_length": 20000,  # maximum segment length
    "sequence_length": 20,  # consider historical sequences for RNN models
    "delta_t": 0.01,  # update delta t for dynamic models
    "delay_steps": 6,  # consider model delays between input and output
    "window_size": 51,  # window_size for savgol_filter
    "polynomial_order": 3  # polynomial_order for savgol_filter
}

point_mass_config = {
    "calibration_dimension": 3,  # Calibration (speed, command, acceleration)
    "vehicle_model": "dev_kit_v3"
}

lstm_model_config = {
    "fnn_layers": 3,  # depth of the feed-forward neural nets
    "epochs": 30  # training epochs
}

mlp_model_config = {
    "fnn_layers": 2,  # depth of the feed-forward neural nets
    "epochs": 30  # training epochs
}

imu_scaling = {
    # IMU compensation for localization scaling issue
    "pp6": 0.55,  # scaling rate for acceleration and angular speed on pp6
    "pp7": 1.085  # scaling rate for acceleration and angular speed on pp7
}

acc_method = {
    "acc_from_IMU": True,  # getting acceleration from differential of localization
    "acc_from_speed": False,  # getting acceleration from differential of speed
    "add_smooth_to_speed": False,  # smooth speed before differential
    "plot_model": False
}

feature_extraction = {
    "inter_result_folder": "modules/control/tmp/dynamic_model",  # intermediate result folder
    "output_folder": "modules/control/result/dynamic_model",  # final result folder
    "uniform_output_folder": "modules/control/tmp/uniform",  # uniform distributed data set
    "incremental_process": False,  # turn on incremental data processing
    "gear": 2,  # 1: gear_drive, 2:gear_reverse
    "sample_size": 6000  # 200 for local test
}
