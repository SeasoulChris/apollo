#!/usr/bin/env python

import time
import casadi as ca
import numpy as np
import math

import fueling.common.logging as logging


class TrajectoryPerturbationSynthesizer(object):
    '''
    a synthesizer using optimiaztion to perturbate input trajectory
    '''

    def __init__(self):
        self.splitting_idx = None
        self.perturbate_point_idx = None
        self.perturbate_xy_range = [-0.5, 0.5]
        self.perturbate_heading_range = [-np.pi / 3, np.pi / 3]

        self.ref_cost = 1.0
        self.elastic_band_smoothing_cost = 10.0

        self.max_curvature = 0.2

        np.random.seed(0)

    def perturbate_point(self, trajectory, perturbation_point_idx):
        original_x = trajectory[perturbation_point_idx][0]
        original_y = trajectory[perturbation_point_idx][1]
        original_heading = trajectory[perturbation_point_idx][2]
        x_perturbation = np.random.uniform(low=self.perturbate_xy_range[0],
                                           high=self.perturbate_xy_range[1])
        y_perturbation = np.random.uniform(low=self.perturbate_xy_range[0],
                                           high=self.perturbate_xy_range[1])
        heading_perturbation = np.random.uniform(low=self.perturbate_heading_range[0],
                                                 high=self.perturbate_heading_range[1])
        trajectory[perturbation_point_idx] = np.array([original_x + x_perturbation,
                                                       original_y + y_perturbation,
                                                       original_heading + heading_perturbation])
        return trajectory

    def ebs_with_fixed_point(self, trajectory, fixed_point_idx_list):
        state_horizon = trajectory.shape[0]
        x = ca.MX.sym('x', 1, state_horizon)
        y = ca.MX.sym('y', 1, state_horizon)

        ref_objective = 0
        smoothing_objective = 0

        for i in range(state_horizon):
            ref_objective += (x[i] - trajectory[i][0])**2 + \
                (y[i] - trajectory[i][1])**2

        for i in range(1, state_horizon - 1):
            smoothing_objective += (x[i] - (x[i - 1] + x[i + 1]) / 2)**2 + \
                (y[i] - (y[i - 1] + y[i + 1]) / 2)**2

        objective = self.ref_cost * ref_objective + \
            self.elastic_band_smoothing_cost * smoothing_objective

        lbx_x = ca.DM(1, state_horizon)
        ubx_x = ca.DM(1, state_horizon)
        lbx_y = ca.DM(1, state_horizon)
        ubx_y = ca.DM(1, state_horizon)

        for i in range(state_horizon):
            if i in fixed_point_idx_list:
                lbx_x[i] = trajectory[i][0]
                ubx_x[i] = trajectory[i][0]
                lbx_y[i] = trajectory[i][1]
                ubx_y[i] = trajectory[i][1]
                continue
            lbx_x[i] = -ca.inf
            ubx_x[i] = ca.inf
            lbx_y[i] = -ca.inf
            ubx_y[i] = ca.inf

        lbx = ca.horzcat(lbx_x, lbx_y)
        ubx = ca.horzcat(ubx_x, ubx_y)

        qp = {'x': ca.horzcat(x, y), 'f': objective}
        qp_problem = ca.qpsol('smoothing', 'osqp', qp)
        solution = qp_problem(lbx=lbx, ubx=ubx)

        smoothed_xy = np.array(solution['x'])
        smoothed_xy = smoothed_xy.reshape((2, -1)).T

        return smoothed_xy

    def evaluate_trajectory_heading(self, xy_point, original_trajectory):
        if len(original_trajectory) < 2:
            return original_trajectory
        state_horizon = original_trajectory.shape[0]
        headings = np.zeros((state_horizon, 1))
        headings[0] = math.atan2(original_trajectory[1][1] - original_trajectory[0][1],
                                 original_trajectory[1][0] - original_trajectory[0][0])
        headings[-1] = math.atan2(original_trajectory[-1][1] - original_trajectory[-2][1],
                                  original_trajectory[-1][0] - original_trajectory[-2][0])
        for i in range(1, state_horizon - 1):
            headings[i] = math.atan2(original_trajectory[i + 1][1] - original_trajectory[i - 1][1],
                                     original_trajectory[i + 1][0] - original_trajectory[i - 1][0])
        trajectory = np.hstack((xy_point, headings))
        return trajectory

    def check_trajectory_curvature(self, smoothed_trajectory):
        for i in range(smoothed_trajectory.shape[0] - 1):
            point_distance = math.sqrt((smoothed_trajectory[i][0]
                                        - smoothed_trajectory[i + 1][0])**2
                                       + (smoothed_trajectory[i][1]
                                          - smoothed_trajectory[i + 1][1])**2)
            angle_difference = math.fabs(
                smoothed_trajectory[i + 1][2] - smoothed_trajectory[i][2])
            minimal_angle_difference = (
                angle_difference + np.pi / 2) % np.pi - np.pi / 2
            estimated_curvature = minimal_angle_difference / point_distance
            if estimated_curvature > self.max_curvature:
                return False
        return True

    def visualize_for_debug(self, past_trajectory, future_trajectory,
                            perturbated_past_trajectory, perturbated_future_trajectory):
        origin_traj_for_plot = np.vstack(
            (past_trajectory, future_trajectory))
        traj_for_plot = np.vstack(
            (perturbated_past_trajectory, perturbated_future_trajectory))

        import matplotlib.pyplot as plt
        fig = plt.figure(0)
        xy_graph = fig.add_subplot(111)
        xy_graph.plot(
            origin_traj_for_plot[:, 0], origin_traj_for_plot[:, 1],
            linestyle='--', marker='o', color='r')
        xy_graph.plot(traj_for_plot[:, 0], traj_for_plot[:, 1],
                      linestyle='--', marker='o', color='g')
        original_point = xy_graph.scatter(origin_traj_for_plot[self.perturbate_point_idx][0],
                                          origin_traj_for_plot[self.perturbate_point_idx][1],
                                          marker='D', color='r')
        perturbated_point = xy_graph.scatter(traj_for_plot[self.perturbate_point_idx][0],
                                             traj_for_plot[self.perturbate_point_idx][1],
                                             marker='D', color='g')

        xy_graph.set_aspect('equal')
        plt.show()

    def synthesize_perturbation(self, past_trajectory, future_trajectory):
        '''
        the function to perturbate a point in given trajectories and return smoothed trajectory

        Arguments:
        past_trajectory: (N_past, 3) numpy array including state (x, y, heading)
                        in time ascending order
        future_trajectory: (N_future, 3) numpy array including state (x, y, heading)
                        in time ascending order

        Return:
        perturbated_past_trajectory: (N_past, 3) numpy array including state (x, y, heading)
        perturbated_future_trajectory: (N_future, 3) numpy array including state (x, y, heading)
        '''
        merged_trajectory = np.vstack(
            (past_trajectory, future_trajectory))

        self.splitting_idx = past_trajectory.shape[0] - 1
        self.perturbate_point_idx = np.random.randint(
            self.splitting_idx, merged_trajectory.shape[0] - 1)

        perturbated_trajectory = self.perturbate_point(merged_trajectory,
                                                       self.perturbate_point_idx)

        normalization_point = np.copy(perturbated_trajectory[0, :])

        perturbated_trajectory -= normalization_point

        smoothed_xy = self.ebs_with_fixed_point(perturbated_trajectory,
                                                [0,
                                                 self.perturbate_point_idx,
                                                 merged_trajectory.shape[0] - 1])

        smoothed_trajectory = self.evaluate_trajectory_heading(smoothed_xy,
                                                               perturbated_trajectory)
        smoothed_trajectory += normalization_point

        is_valid = self.check_trajectory_curvature(smoothed_trajectory)

        perturbated_past_trajectory = smoothed_trajectory[:self.splitting_idx + 1]
        perturbated_future_trajectory = smoothed_trajectory[self.splitting_idx + 1:]

        # self.visualize_for_debug(past_trajectory, future_trajectory,
        #                          perturbated_past_trajectory, perturbated_future_trajectory)

        return is_valid, perturbated_past_trajectory, perturbated_future_trajectory


if __name__ == "__main__":
    trajectory = np.zeros((0, 3))
    for i in range(20):
        point = np.array([[i, 0, 0]])
        trajectory = np.vstack((trajectory, point))

    synthesizer = TrajectoryPerturbationSynthesizer()

    is_valid, perturbated_past_trajectory, perturbated_future_trajectory = \
        synthesizer.synthesize_perturbation(trajectory[:10], trajectory[10:])

    traj_for_plot = np.vstack(
        (perturbated_past_trajectory, perturbated_future_trajectory))

    print(is_valid)

    import matplotlib.pyplot as plt
    fig = plt.figure(0)
    xy_graph = fig.add_subplot(111)
    xy_graph.plot(traj_for_plot[:, 0], traj_for_plot[:, 1])
    xy_graph.set_aspect('equal')
    plt.show()
