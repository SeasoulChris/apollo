import torch


def rasterize_vehicle_box(pred_traj_x, pred_traj_y, box_heading,
                          heading_offset,
                          initial_box_x_idx,
                          initial_box_y_idx,
                          center_shift_distance,
                          img_resolution,
                          half_box_length, half_box_width, idx_mesh):
    rear_to_initial_box_x_delta = (torch.cos(heading_offset) * (-pred_traj_y)
                                   - torch.sin(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)
    rear_to_initial_box_y_delta = (torch.sin(heading_offset) * (-pred_traj_y)
                                   + torch.cos(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)

    center_to_rear_x_delta = torch.cos(
        box_heading) * center_shift_distance
    center_to_rear_y_delta = torch.sin(
        box_heading) * center_shift_distance

    center_to_initial_box_x_delta = center_to_rear_x_delta + rear_to_initial_box_x_delta
    center_to_initial_box_y_delta = center_to_rear_y_delta + rear_to_initial_box_y_delta

    mid_center_x_idx = (
        initial_box_x_idx + torch.div(center_to_initial_box_x_delta,
                                      img_resolution))
    mid_center_y_idx = (
        initial_box_y_idx + torch.div(center_to_initial_box_y_delta,
                                      img_resolution))

    delta_x = idx_mesh[:, :, :, 1] - mid_center_x_idx
    delta_y = idx_mesh[:, :, :, 0] - mid_center_y_idx
    abs_transformed_delta_x = torch.abs(torch.cos(-box_heading) * delta_x
                                        - torch.sin(-box_heading) * delta_y)
    abs_transformed_delta_y = torch.abs(torch.sin(-box_heading) * delta_x
                                        + torch.cos(-box_heading) * delta_y)
    return torch.logical_and(abs_transformed_delta_x < half_box_length,
                             abs_transformed_delta_y < half_box_width).float()


def rasterize_vehicle_three_circles_guassian(pred_traj_x,
                                             pred_traj_y,
                                             box_heading,
                                             heading_offset,
                                             initial_box_x_idx,
                                             initial_box_y_idx,
                                             center_shift_distance,
                                             front_shift_distance,
                                             img_resolution,
                                             theta,
                                             sigma_x,
                                             sigma_y,
                                             idx_mesh,
                                             front_center_guassian,
                                             mid_center_guassian,
                                             rear_center_guassian):
    rear_to_initial_box_x_delta = (torch.cos(heading_offset) * (-pred_traj_y)
                                   - torch.sin(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)
    rear_to_initial_box_y_delta = (torch.sin(heading_offset) * (-pred_traj_y)
                                   + torch.cos(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)

    center_to_rear_x_delta = torch.cos(
        box_heading) * center_shift_distance
    center_to_rear_y_delta = torch.sin(
        box_heading) * center_shift_distance

    center_to_initial_box_x_delta = center_to_rear_x_delta + rear_to_initial_box_x_delta
    center_to_initial_box_y_delta = center_to_rear_y_delta + rear_to_initial_box_y_delta

    rear_center_x_idx = (
        initial_box_x_idx + torch.div(rear_to_initial_box_x_delta,
                                      img_resolution))
    rear_center_y_idx = (
        initial_box_y_idx + torch.div(rear_to_initial_box_y_delta,
                                      img_resolution))

    mid_center_x_idx = (
        initial_box_x_idx + torch.div(center_to_initial_box_x_delta,
                                      img_resolution))
    mid_center_y_idx = (
        initial_box_y_idx + torch.div(center_to_initial_box_y_delta,
                                      img_resolution))

    center_to_front_x_delta = torch.cos(
        box_heading) * front_shift_distance
    center_to_front_y_delta = torch.sin(
        box_heading) * front_shift_distance

    front_to_initial_box_x_delta = center_to_front_x_delta + rear_to_initial_box_x_delta
    front_to_initial_box_y_delta = center_to_front_y_delta + rear_to_initial_box_y_delta

    front_center_x_idx = (
        initial_box_x_idx + torch.div(front_to_initial_box_x_delta,
                                      img_resolution))
    front_center_y_idx = (
        initial_box_y_idx + torch.div(front_to_initial_box_y_delta,
                                      img_resolution))

    a = torch.pow(torch.cos(theta), 2) / (2 * torch.pow(sigma_x, 2)) + \
        torch.pow(torch.sin(theta), 2) / (2 * torch.pow(sigma_y, 2))
    b = torch.sin(2 * theta) / (4 * torch.pow(sigma_x, 2)) + \
        torch.sin(2 * theta) / (4 * torch.pow(sigma_y, 2))
    c = torch.pow(torch.sin(theta), 2) / (2 * torch.pow(sigma_x, 2)) + \
        torch.pow(torch.cos(theta), 2) / (2 * torch.pow(sigma_y, 2))
    x_coords = idx_mesh[:, :, :, 1]
    y_coords = idx_mesh[:, :, :, 0]
    front_center_guassian = torch.exp(-(a * torch.pow((x_coords - front_center_x_idx), 2)
                                        + 2 * b
                                        * (x_coords - front_center_x_idx)
                                        * (y_coords - front_center_y_idx)
                                        + c * torch.pow((y_coords - front_center_y_idx), 2)))

    mid_center_guassian = torch.exp(-(a * torch.pow((x_coords - mid_center_x_idx), 2)
                                      + 2 * b * (x_coords - mid_center_x_idx)
                                      * (y_coords - mid_center_y_idx)
                                      + c * torch.pow((y_coords - mid_center_y_idx), 2)))

    rear_center_guassian = torch.exp(-(a * torch.pow((x_coords - rear_center_x_idx), 2)
                                       + 2 * b * (x_coords - rear_center_x_idx)
                                       * (y_coords - rear_center_y_idx)
                                       + c * torch.pow((y_coords - rear_center_y_idx), 2)))
    max_prob = 1.0010
    return (front_center_guassian + mid_center_guassian + rear_center_guassian) / max_prob
