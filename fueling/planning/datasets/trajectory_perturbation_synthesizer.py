#!/usr/bin/env python

import math

from absl import flags
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from fueling.planning.math_utils.math_utils import NormalizeAngle

flags.DEFINE_multi_float('perturbate_normal_direction_range', [-2, 2],
                         'perturbate_normal_direction_range')
flags.DEFINE_multi_float('perturbate_tangential_direction_range', [0.2, 2.0],
                         'perturbate_tangential_direction_range')
flags.DEFINE_bool('is_lateral_or_longitudinal_synthesizing', True,
                  'True for doing lateral synthesizing, '
                  'False for longitudinal synthesizing')
flags.DEFINE_float(
    'ref_cost', 1.0, 'cost weighting on original line')
flags.DEFINE_float('elastic_band_smoothing_cost', 10.0,
                   'smoothing weighting')
flags.DEFINE_float('max_curvature', 0.3,
                   'maximum allowed curvature')


class TrajectoryPerturbationSynthesizer(object):
    '''
    a synthesizer using optimiaztion to perturbate input trajectory
    '''

    def __init__(self, perturbate_normal_direction_range, perturbate_tangential_direction_range,
                 is_lateral_or_longitudinal_synthesizing,
                 ref_cost, elastic_band_smoothing_cost, max_curvature):
        self.perturbate_normal_direction_range = perturbate_normal_direction_range
        self.perturbate_tangential_direction_range = perturbate_tangential_direction_range
        self.is_lateral_or_longitudinal_synthesizing = is_lateral_or_longitudinal_synthesizing
        self.ref_cost = ref_cost
        self.elastic_band_smoothing_cost = elastic_band_smoothing_cost
        self.max_curvature = max_curvature

    def lateral_perturbate_point(self, trajectory, perturbation_point_idx):
        original_x = trajectory[perturbation_point_idx][0]
        original_y = trajectory[perturbation_point_idx][1]
        perturbate_normal_direction_range = \
            np.random.uniform(low=self.perturbate_normal_direction_range[0],
                              high=self.perturbate_normal_direction_range[1])
        normal_vectors = NormalizeAngle(
            trajectory[perturbation_point_idx][2] + math.pi / 2)
        trajectory[perturbation_point_idx][:2] = np.array([original_x + np.cos(normal_vectors)
                                                           * perturbate_normal_direction_range,
                                                           original_y
                                                           + np.sin(normal_vectors)
                                                           * perturbate_normal_direction_range])
        return trajectory

    def longitudinal_perturbate_point(self, trajectory, cur_point_idx):
        longitudinal_shifting_vec_perturbate_ratio = \
            np.random.uniform(low=self.perturbate_tangential_direction_range[0],
                              high=self.perturbate_tangential_direction_range[1])

        longitudinal_perturbate_trajectory = np.copy(trajectory)
        for i in range(cur_point_idx + 1, trajectory.shape[0]):
            cur_point_x = longitudinal_perturbate_trajectory[i][0]
            cur_point_y = longitudinal_perturbate_trajectory[i][1]
            past_point_x = longitudinal_perturbate_trajectory[i - 1][0]
            past_point_y = longitudinal_perturbate_trajectory[i - 1][1]
            longitudinal_perturbate_trajectory[i][:2] = np.array(
                [past_point_x + (cur_point_x - past_point_x)
                    * longitudinal_shifting_vec_perturbate_ratio,
                 past_point_y + (cur_point_y - past_point_y)
                    * longitudinal_shifting_vec_perturbate_ratio])

        return longitudinal_perturbate_trajectory

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
        opt_dict = {'osqp.verbose': False}
        qp_problem = ca.qpsol('smoothing', 'osqp', qp, opt_dict)
        solution = qp_problem(lbx=lbx, ubx=ubx)

        smoothed_xy = np.array(solution['x'])
        smoothed_xy = smoothed_xy.reshape((2, -1)).T

        return smoothed_xy

    def evaluate_trajectory_heading(self, xy_point):
        state_horizon = xy_point.shape[0]
        if state_horizon < 2:
            return xy_point, False
        headings = np.zeros((state_horizon, 1))
        headings[0] = math.atan2(xy_point[1][1] - xy_point[0][1],
                                 xy_point[1][0] - xy_point[0][0])
        headings[-1] = math.atan2(xy_point[-1][1] - xy_point[-2][1],
                                  xy_point[-1][0] - xy_point[-2][0])
        for i in range(1, state_horizon - 1):
            headings[i] = math.atan2(xy_point[i + 1][1] - xy_point[i - 1][1],
                                     xy_point[i + 1][0] - xy_point[i - 1][0])
        return np.hstack((xy_point, headings)), True

    def check_trajectory_curvature(self, smoothed_trajectory):
        for i in range(smoothed_trajectory.shape[0] - 1):
            point_distance = math.sqrt((smoothed_trajectory[i][0]
                                        - smoothed_trajectory[i + 1][0])**2
                                       + (smoothed_trajectory[i][1]
                                          - smoothed_trajectory[i + 1][1])**2)
            angle_difference = math.fabs(
                NormalizeAngle(smoothed_trajectory[i + 1][2] - smoothed_trajectory[i][2]))
            estimated_curvature = angle_difference / point_distance
            if estimated_curvature > self.max_curvature:
                return False
        return True

    def visualize_for_debug(self, frame_file_path, past_trajectory, future_trajectory,
                            perturbated_past_trajectory, perturbated_future_trajectory,
                            perturbate_point_idx):
        origin_traj_for_plot = np.vstack(
            (past_trajectory, future_trajectory))
        traj_for_plot = np.vstack(
            (perturbated_past_trajectory, perturbated_future_trajectory))
        canvas = plt.figure()
        fig = canvas.add_subplot(1, 1, 1)
        fig.plot(
            origin_traj_for_plot[:, 0], origin_traj_for_plot[:, 1],
            linestyle='--', marker='o', color='r')
        fig.plot(traj_for_plot[:, 0], traj_for_plot[:, 1],
                 linestyle='--', marker='o', color='g')
        fig.scatter(origin_traj_for_plot[perturbate_point_idx][0],
                    origin_traj_for_plot[perturbate_point_idx][1],
                    marker='D', color='r')
        fig.scatter(traj_for_plot[perturbate_point_idx][0],
                    traj_for_plot[perturbate_point_idx][1],
                    marker='D', color='g')
        for i in range(traj_for_plot.shape[0]):
            fig.arrow(traj_for_plot[i, 0], traj_for_plot[i, 1], np.cos(
                traj_for_plot[i, 2]), np.sin(traj_for_plot[i, 2]), color='g')
            fig.arrow(origin_traj_for_plot[i, 0], origin_traj_for_plot[i, 1], np.cos(
                origin_traj_for_plot[i, 2]), np.sin(origin_traj_for_plot[i, 2]), color='r')

        fig.set_aspect('equal')
        canvas.savefig(frame_file_path)
        plt.close(canvas)

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

        splitting_idx = past_trajectory.shape[0] - 1

        if self.is_lateral_or_longitudinal_synthesizing:
            perturbate_point_idx = np.random.randint(
                splitting_idx, merged_trajectory.shape[0] - 1)

            perturbated_trajectory = self.lateral_perturbate_point(merged_trajectory,
                                                                   perturbate_point_idx)

            normalization_point = np.copy(perturbated_trajectory[0, :])

            perturbated_trajectory[:, :2] -= normalization_point[:2]

            smoothed_xy = self.ebs_with_fixed_point(perturbated_trajectory,
                                                    [0,
                                                     perturbate_point_idx,
                                                     merged_trajectory.shape[0] - 1])

            smoothed_trajectory, is_long_enough = self.evaluate_trajectory_heading(
                smoothed_xy)

            smoothed_trajectory[:, :2] += normalization_point[:2]
        else:
            perturbate_point_idx = splitting_idx

            perturbated_trajectory = self.longitudinal_perturbate_point(merged_trajectory,
                                                                        perturbate_point_idx)
            smoothed_trajectory, is_long_enough = self.evaluate_trajectory_heading(
                perturbated_trajectory)

        is_valid = self.check_trajectory_curvature(smoothed_trajectory)

        perturbated_past_trajectory = smoothed_trajectory[:splitting_idx + 1]
        perturbated_future_trajectory = smoothed_trajectory[splitting_idx + 1:]

        return is_valid and is_long_enough, \
            perturbated_past_trajectory, \
            perturbated_future_trajectory, \
            perturbate_point_idx


if __name__ == "__main__":
    trajectory = np.zeros((0, 3))
    for i in range(20):
        point = np.array([[i, 0, 0]])
        trajectory = np.vstack((trajectory, point))

    synthesizer = TrajectoryPerturbationSynthesizer(perturbate_normal_direction_range=[-1.5, 1.5],
                                                    perturbate_tangential_direction_range=[
                                                        0.2, 2.0],
                                                    ref_cost=1.0,
                                                    elastic_band_smoothing_cost=10.0,
                                                    max_curvature=0.3)

    is_valid, perturbated_past_trajectory, perturbated_future_trajectory, \
        perturbate_point_idx = \
        synthesizer.synthesize_perturbation(trajectory[:10], trajectory[10:])

    traj_for_plot = np.vstack(
        (perturbated_past_trajectory, perturbated_future_trajectory))

    print(is_valid)

    fig = plt.figure(0)
    xy_graph = fig.add_subplot(111)
    xy_graph.plot(traj_for_plot[:, 0], traj_for_plot[:, 1])
    xy_graph.set_aspect('equal')
    plt.show()
