import torch


def kinematic_action_constraints_layer(max_abs_steering_angle, max_acceleration,
                                       max_deceleration, current_action):
    steering = current_action[:, 0]
    acceleration_range = max_acceleration - max_deceleration
    a = current_action[:, 1]
    constrained_steering = torch.tanh(
        steering) * max_abs_steering_angle
    constrained_a = torch.tanh(a) * 0.5 * acceleration_range \
        + (max_acceleration - 0.5 * acceleration_range)
    return torch.cat((constrained_steering.clone().unsqueeze(1),
                      constrained_a.clone().unsqueeze(1)), dim=1)


def rear_kinematic_model_layer(wheel_base, delta_t, current_states, current_action):
    '''
    current_states: x_0, y_0, phi_0, v_0
    current_action: steering_0, a_0
    return predicted_states
    '''
    x_0 = current_states[:, 0]
    y_0 = current_states[:, 1]
    phi_0 = current_states[:, 2]
    v_0 = current_states[:, 3]
    steering_0 = current_action[:, 0]
    a_0 = current_action[:, 1]

    delta_v = a_0
    delta_phi = (v_0 + 0.5 * delta_v) / \
        wheel_base * torch.tan(steering_0)
    delta_x = (v_0 + 0.5 * delta_v) * torch.cos(phi_0 + 0.5 * delta_phi)
    delta_y = (v_0 + 0.5 * delta_v) * torch.sin(phi_0 + 0.5 * delta_phi)

    # delta_v = a_0
    # delta_phi = v_0 / self.wheel_base * torch.tan(steering_0)
    # delta_x = v_0 * torch.cos(phi_0)
    # delta_y = v_0 * torch.sin(phi_0)

    x_1 = delta_x * delta_t + x_0
    y_1 = delta_y * delta_t + y_0
    phi_1 = delta_phi * delta_t + phi_0
    v_1 = delta_v * delta_t + v_0

    return torch.cat((x_1.unsqueeze(1),
                      y_1.unsqueeze(1),
                      phi_1.unsqueeze(1),
                      v_1.unsqueeze(1)), dim=1)


def center_kinematic_model_layer(lr, lf, delta_t, current_states, current_action):
    '''
    current_states: x_0, y_0, phi_0, v_0
    current_action: steering_0, a_0
    return predicted_states
    '''
    x_0 = current_states[:, 0]
    y_0 = current_states[:, 1]
    phi_0 = current_states[:, 2]
    v_0 = current_states[:, 3]
    steering_0 = current_action[:, 0]
    a_0 = current_action[:, 1]

    beta = torch.atan(lr / (lr + lf)
                      * torch.tan(steering_0))
    delta_x = v_0 * torch.cos(phi_0 + beta)
    delta_y = v_0 * torch.sin(phi_0 + beta)
    delta_phi = v_0 / lr * torch.sin(beta)
    delta_v = a_0

    x_1 = delta_x * delta_t + x_0
    y_1 = delta_y * delta_t + y_0
    phi_1 = delta_phi * delta_t + phi_0
    v_1 = delta_v * delta_t + v_0

    return torch.cat((x_1.unsqueeze(1),
                      y_1.unsqueeze(1),
                      phi_1.unsqueeze(1),
                      v_1.unsqueeze(1)), dim=1)
