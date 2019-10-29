#!/usr/bin/env python

import sys
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cyber_py import cyber
from modules.localization.proto import localization_pb2

from fueling.planning.stability.grading import stability_grader
from fueling.planning.stability.libs.imu_angular_velocity import ImuAngularVelocity
from fueling.planning.stability.libs.imu_speed_jerk import ImuSpeedJerk


LAT_STABILITY_SCORE = []
LAT_STABILITY_TIME = []

LON_STABILITY_SCORE = []
LON_STABILITY_TIME = []

begin_t = None
last_t = None
lock = threading.Lock()

PLOT_DATA_LENGTH = 500

lat_jerk_processor = ImuSpeedJerk(is_lateral=True)
lon_jerk_processor = ImuSpeedJerk(is_lateral=False)
av_processor = ImuAngularVelocity()
grader = stability_grader.Grader()


def callback(pose_pb):
    global LAT_STABILITY_SCORE, LAT_STABILITY_TIME
    global LON_STABILITY_SCORE, LON_STABILITY_TIME
    global lat_jerk_processor, lon_jerk_processor, av_processor
    global begin_t, last_t

    lock.acquire()

    if begin_t is None:
        begin_t = pose_pb.header.timestamp_sec
    current_t = pose_pb.header.timestamp_sec
    if last_t is not None and abs(current_t - last_t) > 1:
        begin_t = pose_pb.header.timestamp_sec
        lat_jerk_processor = ImuSpeedJerk(is_lateral=True)
        lon_jerk_processor = ImuSpeedJerk(is_lateral=False)
        av_processor = ImuAngularVelocity()
        LAT_STABILITY_SCORE = []
        LAT_STABILITY_TIME = []
        LON_STABILITY_SCORE = []
        LON_STABILITY_TIME = []

    av_processor.add(pose_pb)
    lat_jerk_processor.add(pose_pb)
    lon_jerk_processor.add(pose_pb)

    av = av_processor.get_latest_corrected_angular_velocity()
    lat_jerk = lat_jerk_processor.get_lastest_jerk()
    lon_jerk = lon_jerk_processor.get_lastest_jerk()

    if av is not None and lat_jerk is not None and lon_jerk is not None:
        rs = grader.grade(lat_jerk, lon_jerk, av)

        LAT_STABILITY_TIME.append(current_t - begin_t)
        if rs == 0:
            rs = 0.000000000000001
        LAT_STABILITY_SCORE.append(math.log10(rs))

    lock.release()

    last_t = current_t


def listener():
    cyber.init()
    test_node = cyber.Node("pose_listener")
    test_node.create_reader("/apollo/localization/pose",
                            localization_pb2.LocalizationEstimate, callback)


def compensate(data_list):
    comp_data = [0] * PLOT_DATA_LENGTH
    comp_data.extend(data_list)
    if len(comp_data) > PLOT_DATA_LENGTH:
        comp_data = comp_data[-PLOT_DATA_LENGTH:]
    return comp_data


def update(frame_number):
    lock.acquire()
    lat_stability_line.set_xdata(LAT_STABILITY_TIME)
    lat_stability_line.set_ydata(LAT_STABILITY_SCORE)

    lon_stability_line.set_xdata(LON_STABILITY_TIME)
    lon_stability_line.set_ydata(LON_STABILITY_SCORE)

    lock.release()
    if len(LAT_STABILITY_SCORE) > 0:
        lat_stability_text.set_text('init point v = %.1f' % LAT_STABILITY_SCORE[-1])


if __name__ == '__main__':
    argv = FLAGS(sys.argv)
    listener()
    fig, ax = plt.subplots()
    X = range(PLOT_DATA_LENGTH)
    Xs = sorted([-i for i in X])
    lat_stability_line, = ax.plot(
        LAT_STABILITY_TIME, LAT_STABILITY_SCORE, 'b.', lw=2, alpha=0.5, label='lat stability')

    lon_stability_line, = ax.plot(
        LON_STABILITY_TIME, LON_STABILITY_SCORE, 'r.', lw=2, alpha=0.5, label='lon stability')

    lat_stability_text = ax.text(0.75, 0.95, '', transform=ax.transAxes)

    ani = animation.FuncAnimation(fig, update, interval=100)
    ax.set_ylim(-20, 5)
    ax.set_xlim(-1, 60)
    ax.legend(loc="upper left")
    plt.show()
