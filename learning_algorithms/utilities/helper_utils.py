###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import numpy as np


def world_coord_to_relative_coord(input_world_coord, ref_world_coord):
    x_diff = input_world_coord[0] - ref_world_coord[0]
    y_diff = input_world_coord[1] - ref_world_coord[1]
    rho = np.sqrt(x_diff ** 2 + y_diff ** 2)
    theta = np.arctan2(y_diff, x_diff) - ref_world_coord[2]

    return (np.cos(theta)*rho, np.sin(theta)*rho)
