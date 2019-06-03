#!/usr/bin/env python
""" Plot the extracted control features in histogram figures """
import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py
import numpy as np


brake_axis_cmd_min = -30.0  # brake_max
brake_axis_cmd_max = -7.0

train_speed_min = 0.0
train_speed_max = 20.0
train_speed_segment = 50
train_cmd_segment = 10


# speed
speed_array = np.linspace(train_speed_min, train_speed_max, num=train_speed_segment)

# cmd
cmd_array = np.linspace(brake_axis_cmd_min, brake_axis_cmd_max, num=train_cmd_segment)

speed_array_brake, cmd_array_brake = np.meshgrid(speed_array, cmd_array)
grid_array_brake = np.array([[s, c] for s, c in zip(
    np.ravel(speed_array_brake), np.ravel(cmd_array_brake))])

file_name = "Transit_brake_calibration_table.pb.txt.hdf5"
f = h5py.File(file_name, 'r+')
names = [n for n in f.keys()]


for i in range(len(names)):
    acc_array_brake = np.array(f[names[i]])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(speed_array_brake, cmd_array_brake, acc_array_brake,
                alpha=1, rstride=1, cstride=1, linewidth=0.5, antialiased=True)
ax.set_xlabel('$speed$')
ax.set_ylabel('$brake$')
ax.set_zlabel('$acceleration$')
plt.show()


throttle_axis_cmd_min = 5.0
throttle_axis_cmd_max = 30.0

fig_throttle = plt.figure()
ax_throttle = fig_throttle.gca(projection='3d')

# cmd
cmd_array_throttle = np.linspace(
    throttle_axis_cmd_min, throttle_axis_cmd_max, num=train_cmd_segment)

file_name_throttle = "Transit_throttle_calibration_table.pb.txt.hdf5"
f_throttle = h5py.File(file_name_throttle, 'r+')
names_throttle = [n for n in f_throttle.keys()]
# print('f.keys', f.keys())
# print('names', names)
for i in range(len(names_throttle)):
    acc_array_throttle = np.array(f_throttle[names_throttle[i]])

speed_array_throttle, cmd_array_throttle = np.meshgrid(speed_array, cmd_array_throttle)
grid_array_throttle = np.array([[s, c] for s, c in zip(
    np.ravel(speed_array), np.ravel(cmd_array_throttle))])

ax_throttle.plot_surface(speed_array_throttle, cmd_array_throttle, acc_array_throttle,
                         alpha=1, rstride=1, cstride=1, linewidth=0.5, antialiased=True)
ax_throttle.set_xlabel('$speed$')
ax_throttle.set_ylabel('$throttle$')
ax_throttle.set_zlabel('$acceleration$')
plt.show()
