#!/usr/bin/env python
"""Utils for localization related calculation."""

import math

from scipy.optimize import fmin
import numpy as np


def get_imu_list_from_pose_list(poses, imu_scaling_acc, imu_scaling_heading_rate):
    """
    Get imu values from localization poses, in imu_scaling_settings there are:
    imu_scaling_acc: scaling rate for acceleration
    imu_scaling_heading_rate: scaling rate for angular speed
    """

    def _normalize_angle(theta):
        """Get normalized angle"""
        theta = theta % (2 * math.pi)
        if theta > math.pi:
            theta = theta - 2 * math.pi
        return theta

    def _distance_s(acc, dt, v):
        """Get distance"""
        return v * dt + 0.5 * acc * dt * dt

    def _calc_theta(poses, data_dim, w, dt):
        """From heading rate to heading"""
        theta = np.zeros((data_dim,))
        init_heading = _normalize_angle(poses[0].heading)
        # Init heading from GPS
        theta[0] = init_heading
        for idx in range(1, data_dim):
            # theta = theta_0 + omega * dt
            theta[idx] = _normalize_angle(init_heading + w[idx - 1] * dt)
            # Update init heading for next step
            init_heading = theta[idx]
        return theta

    def _calc_s(poses, data_dim, acc, dt):
        """Calculate s, s = v0 * dt + 0.5 * a * t * t"""
        # Initial velocity
        init_normalized_heading = _normalize_angle(poses[0].heading)
        v = (poses[0].linear_velocity.x * np.cos(init_normalized_heading)
             + poses[0].linear_velocity.y * np.sin(init_normalized_heading))
        s = np.zeros((data_dim, 1))
        for idx in range(1, data_dim):
            if v + acc[idx - 1] * dt < 0:
                acc[idx - 1] = 0
            s[idx] = _distance_s(acc[idx - 1], dt, v)
            v = v + acc[idx - 1] * dt
        return s

    def _get_location_from_a_and_w(poses, dim, ws, accs):
        """Integration from acceleration and heading angle change rate to x, y"""
        delta_t = 0.01
        theta, s = _calc_theta(poses, dim, ws, delta_t), _calc_s(poses, dim, accs, delta_t)
        localization_x, localization_y = np.zeros((data_dim, 1)), np.zeros((data_dim, 1))
        x_init, y_init = poses[0].position.x, poses[0].position.y
        for idx in range(0, dim):
            localization_x[idx] = x_init + s[idx] * np.cos(theta[idx])
            localization_y[idx] = y_init + s[idx] * np.sin(theta[idx])
            x_init = localization_x[idx]
            y_init = localization_y[idx]
        return (localization_x, localization_y)

    def _get_accelaration(poses, dim):
        """Get accelaration"""
        acc = np.empty((dim, 1))
        for idx, pose in enumerate(poses):
            normalized_heading = _normalize_angle(pose.heading)
            acc[idx] = (pose.linear_acceleration.x * np.cos(normalized_heading)
                        + pose.linear_acceleration.y * np.sin(normalized_heading))
        return acc

    if not poses:
        return None
    data_dim = len(poses)
    imu_acc = imu_scaling_acc * _get_accelaration(poses, data_dim)
    imu_w = imu_scaling_heading_rate * np.array([wz.angular_velocity.z for wz in poses])
    return _get_location_from_a_and_w(poses, data_dim, imu_w, imu_acc)


def find_optimized_imu_scaling(localization_protos):
    """Try composition of acceleration/angular speed values to find out the closest imu scaling"""

    def _compare_poses_imus(imu_scaling_settings, poses):
        """Compare poses and imus by accumulating there diffs"""
        imu_scaling_acc, imu_scaling_heading = imu_scaling_settings[0], imu_scaling_settings[1]
        print(F'imu_scaling_acc: {imu_scaling_acc}, imu_scaling_heading: {imu_scaling_heading}')
        imu_x, imu_y = get_imu_list_from_pose_list(poses, imu_scaling_acc, imu_scaling_heading)
        accumulation = 0
        for idx in range(len(poses)):
            accumulation += (
                (poses[idx].position.x - imu_x[idx]) ** 2
                + (poses[idx].position.y - imu_y[idx]) ** 2)
        return accumulation

    poses = [localization.pose for localization in localization_protos]
    initial_scaling_acc, initial_scaling_heading = 0.5, 0.5
    return fmin(
        _compare_poses_imus, args=(poses,),
        x0=np.array([initial_scaling_acc, initial_scaling_heading]), maxiter=100)
