#!/usr/bin/env python

segment_index = {
    "heading": 0,  # pose.heading
    "q_x": 1,  # pose.orientation.qx
    "q_y": 2,  # pose.orientation.qy
    "q_z": 3, # pose.orientation.qz 
    "q_w": 4, # pose.orientation.qw 
    "v_x": 5,  # pose.linear_velocity.x
    "v_y": 6,  # pose.linear_velocity.y
    "v_z": 7,  # pose.linear_velocity.z
    "a_x": 8,  # pose.linear_acceleration.x
    "a_y": 9,  # pose.linear_acceleration.y
    "a_z": 10,  # pose.linear_acceleration.z
    "w_x": 11,  # pose.angular_velocity.x
    "w_y": 12,  # pose.angular_velocity.y
    "w_z": 13,  # pose.angular_velocity.z
    "speed": 14, # chassis.speed_mps
    "throttle": 15, # chassis.throttle_percentage/100.0
    "brake": 16, # chassis.brake_percentage/100.0
    "steering": 17, # chassis.steering_percentage/100.0
    "mode": 18, # chassis.driving_mode
    "x": 19, # pose.position.x
    "y": 20, # pose.position.y
    "z": 21 # pose.position.z
}

input_index = {
    "speed": 0,  # chassis.speed_mps
    "acceleration": 1,  # a_x * cos(heading) + a_y * sin(heading)
    "throttle": 2,  # chassis.throttle_percentage/100.0
    "brake": 3,  # chassis.brake_percentage/100.0
    "steering": 4  # chassis.steering_percentage/100.
}

output_index = {
    "acceleration": 0,  # a_x * cos(heading) + a_y * sin(heading)
    "w_z": 1  # pose.angular_velocity.z
}

feature_config = {
    "input_dim": 5, # input feature dimension
    "output_dim": 2, # output feature dimension
    "maximum_segment_length": 20000, # maximum segment length
    "sequence_length": 20, # consider historical sequences for RNN models
    "delta_t": 0.01, # update delta t for dynamic models
    "delay_steps": 1, # consider model delays between input and output
    "window_size": 51, # window_size for savgol_filter
    "polynomial_order": 3 # polynomial_order for savgol_filter
}

point_mass_config = {
    "calibration_dimension": 3,  # Calibration (speed, command, acceleration)
    "vehicle_model": "mkz7"
}

lstm_model_config = {
    "fnn_layers": 2, # depth of the feed-forward neural nets
    "epochs": 10 # training epochs
}

mlp_model_config = {
    "fnn_layers": 2, # depth of the feed-forward neural nets
    "epochs": 10 # training epochs
}
