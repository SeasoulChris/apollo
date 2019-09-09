#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""Utils for displaying."""
import matplotlib

matplotlib.use('Agg')

import datetime
import math
import pytz
import sys

import matplotlib.pyplot as plt
import mpld3

from modules.common.proto.drive_event_pb2 import DriveEvent

TIMEZONE = 'America/Los_Angeles'


def timestamp_to_time(timestamp):
    """Convert Unix epoch timestamp to readable time."""
    dt = datetime.datetime.fromtimestamp(timestamp, pytz.utc)
    local_tz = pytz.timezone(TIMEZONE)
    return dt.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')


def timestamp_ns_to_time(timestamp):
    """Convert Unix epoch timestamp to readable time."""
    return timestamp_to_time(timestamp / 1e9)


def draw_path_on_gmap(driving_path, canvas_id):
    """Draw driving path on Google map."""
    if not driving_path:
        return ''
    # Get center and zoom.
    min_lat, max_lat = sys.float_info.max, -sys.float_info.max
    min_lng, max_lng = sys.float_info.max, -sys.float_info.max

    for point in driving_path:
        if point.lat > max_lat:
            max_lat = point.lat
        if point.lat < min_lat:
            min_lat = point.lat
        if point.lon > max_lng:
            max_lng = point.lon
        if point.lon < min_lng:
            min_lng = point.lon
    center_lat = (min_lat + max_lat) / 2.0
    center_lng = (min_lng + max_lng) / 2.0
    zoom = int(math.log(1.28 / (max_lat - min_lat + 0.001)) / math.log(2.0)) + 8

    result = 'var gmap = LoadGoogleMap("{}", {}, {}, {});\n'.format(
        canvas_id, center_lat, center_lng, zoom)
    latlng_list = ['[{},{}]'.format(point.lat, point.lon) for point in driving_path]
    result += 'var latlng_list = [{}];\n'.format(','.join(latlng_list))
    result += 'DrawPolyline(gmap, latlng_list, "blue", 2);\n'

    start, end = driving_path[0], driving_path[-1]
    result += 'DrawCircle(gmap, {}, {}, 20, "green");\n'.format(start.lat, start.lon)
    result += 'DrawCircle(gmap, {}, {}, 20, "red");\n'.format(end.lat, end.lon)

    return result


def draw_disengagements_on_gmap(record):
    """Draw disengagements on Google map."""
    result = ''
    for dis in record.disengagements:
        info = 'disengage at %.1fs' % (dis.time - record.header.begin_time / 1e9)
        result += 'DrawInfoWindow(gmap, {}, {}, "{}");\n'.format(
            dis.location.lat, dis.location.lon, info)
    return result


def readable_data_size(num):
    """Print data size in readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']:
        if num < 1024.0:
            return '%.2f %s' % (num, unit)
        num /= 1024.0
    return "%.2f %s" % (num, 'YB')


def meter_to_miles(meters):
    """Convert meter to miles."""
    return "%.2f" % (meters / 1609.344)


def drive_event_type_name(event_type):
    """Convert DriveEvent type to name."""
    return DriveEvent.Type.Name(event_type)


def plot_record(record):
    """Plot a record as html."""
    # Standard matplotlib plotting, except animation which is not supported.
    # To add new subplot, you may need to:
    # 1. Add necessary fields in fueling/data/proto/record_meta.proto.
    # 2. Extend fueling/data/record_parser.py to extract data and populate the fields.
    # 3. Read the fields here and plot properly.
    fig, axs = plt.subplots(2, 1)
    planning_latency = record.stat.planning_stat.latency.latency_hist
    latency_keys = ["latency_0_10_ms", "latency_20_40_ms", "latency_40_60_ms", "latency_60_80_ms",
                    "latency_80_100_ms", "latency_100_120_ms", "latency_120_150_ms",
                    "latency_150_200_ms", "latency_200_up_ms"]
    latency_values = [planning_latency.get(key, 0) for key in latency_keys]
    axs[0].bar(latency_keys, latency_values)

    stability = record.stat.planning_stat.stability
    speed_x = []
    jerk_y = []
    for speed_jerk in stability.speed_jerk:
        speed = speed_jerk.speed
        for jerk_cnt in speed_jerk.jerk_cnt:
            jerk = jerk_cnt.jerk
            speed_x.append(speed)
            jerk_y.append(jerk)
    axs[1].set_xlabel("speed")
    axs[1].set_ylabel("jerk")
    axs[1].plot(speed_x, jerk_y, 'rx')

    base_speed_jerk = {0: [0, 1, 2, 3, 4], 1: [2, 3, 1, 0, 4, 5], 2: [2, 1, 0, 3, 4, -1], 3: [1, 0, 2, 3, 4, 5, -1, -2],
                       4: [0, -1, 1, 3, 4, 2, -2, 5], 5: [0, 1, 2, 3, -1, 4], 6: [1, 2, 0, 3, -1, 4, 5],
                       7: [2, 1, 0, -1, 3, 4, -2, 5], 8: [0, -1, 1, 2, 3, 4], 9: [0, 1, 2, -1], 10: [0, 1, -1, 2],
                       11: [0, -2, -1, 1, 2], 12: [0, -1, 1, -2, 2, 3], 13: [-1, -2, 0, 1, 2],
                       14: [0, 1, -1, -2, -3, 2, 3], 15: [0, 1, 2, -1, -2], 16: [0, 1, -1, 2, -2], 17: [0, 1, 2],
                       18: [0, 1, 2, -1], 19: [0, 1]}

    speed_x = []
    jerk_y = []
    for speed, jerk_list in base_speed_jerk.items():
        for jerk in jerk_list:
            speed_x.append(speed)
            jerk_y.append(jerk)
    axs[1].plot(speed_x, jerk_y, 'g.')

    # Use mpld3 to transform to HTML.
    # Known issues:
    # 1. Tick labels are not supported: https://github.com/mpld3/mpld3/issues/22
    #
    # You are suggested to test the actual output with `plot_record_test.py`.
    return mpld3.fig_to_html(fig)


# To be registered into jinja2 templates.
utils = {
    'draw_disengagements_on_gmap': draw_disengagements_on_gmap,
    'draw_path_on_gmap': draw_path_on_gmap,
    'readable_data_size': readable_data_size,
    'timestamp_to_time': timestamp_to_time,
    'timestamp_ns_to_time': timestamp_ns_to_time,
    'drive_event_type_name': drive_event_type_name,
    'meter_to_miles': meter_to_miles,
    'plot_record': plot_record,
}
