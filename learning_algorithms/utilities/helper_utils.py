#!/usr/bin/env python
"""Helper utils."""

import numpy as np


def world_coord_to_relative_coord(input_world_coord, ref_world_coord):
    x_diff = input_world_coord[0] - ref_world_coord[0]
    y_diff = input_world_coord[1] - ref_world_coord[1]
    rho = np.sqrt(x_diff ** 2 + y_diff ** 2)
    theta = np.arctan2(y_diff, x_diff) - ref_world_coord[2]

    return (np.cos(theta)*rho, np.sin(theta)*rho)
