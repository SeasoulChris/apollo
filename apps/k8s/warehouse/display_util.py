#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""Utils for displaying."""
import matplotlib

matplotlib.use('Agg')

from collections import Counter
import datetime
import math
import pytz
import sys

import matplotlib.pyplot as plt
import mpld3

from modules.common.proto.drive_event_pb2 import DriveEvent
import fueling.common.logging as logging
import fueling.common.redis_utils as redis_utils


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
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10, 15)
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

    # lat speed jerk
    speed_x = []
    jerk_y = []
    for speed_jerk in stability.speed_jerk:
        speed = speed_jerk.speed
        for jerk_cnt in speed_jerk.jerk_cnt:
            jerk = jerk_cnt.jerk
            speed_x.append(speed)
            jerk_y.append(jerk)
    axs[2].set_xlabel("speed")
    axs[2].set_ylabel("jerk")
    axs[2].plot(speed_x, jerk_y, 'rx')

    base_speed_jerk = {0: [0, 1, 2, 3, -1, -2, -3, -5, -4, 4, 5], 1: [3, 4, 2, 1, -1, -2, -3, 0, -4, -5, -6],
                       2: [1, 0, -1, -2, -5, -6, -4, -3, 2, 3], 3: [0, -1, 1, -2, -3, -4, -5, 2, 3, 4, 5, -6, -7],
                       4: [0, -1, 1, 2, 3, -2, -3, 4, -4], 5: [0, -1, 1, 2, 3, 4, -2, -3],
                       6: [0, 1, -1, -2, 2, -3, -4, 3], 7: [0, 1, 2, 3, 4, -1, -2, -3], 8: [0, -1, 1, 2, 3, 4, -2],
                       9: [0, -1, 1], 10: [0, -1, 1, 2, -2], 11: [0, -1, -2, 1], 12: [0, -1, -2], 13: [-1, -2, 0],
                       -11: [0, 1, 3, 2], -18: [0, 1], -17: [0, 1, -1, 2], -16: [0, 1, -1, 2], -15: [0, 1, 2, 3],
                       -14: [0, -1, 1, 2, 3, 4], -13: [0, 1, 2, 3, -1], -12: [0, 1, -1, 3, 2],
                       -1: [-1, 0, 1, 3, 4, 2, -2, -3, -4, 5, 6], -10: [0, 1, -1, 2, 3], -9: [0, 1, -1, 2],
                       -8: [0, 1, 2, 5, 4, 3], -7: [0, 1, 4, 3, 5, 2], -6: [0, 1, 2, 4, 5, 3, -1],
                       -5: [1, 0, 3, 2, -1, 4, 5], -4: [3, 2, 4, 1, 0, -1], -3: [1, 0, 2, 3, -1, 4, 5, 6],
                       -2: [0, 1, 2, 3, 4, 5, -1, -3, -2, 6]}

    speed_x = []
    jerk_y = []
    for speed, jerk_list in base_speed_jerk.items():
        for jerk in jerk_list:
            speed_x.append(speed)
            jerk_y.append(jerk)
    axs[2].plot(speed_x, jerk_y, 'g.')

    # Use mpld3 to transform to HTML.
    # Known issues:
    # 1. Tick labels are not supported: https://github.com/mpld3/mpld3/issues/22
    #
    # You are suggested to test the actual output with `plot_record_test.py`.
    return mpld3.fig_to_html(fig)


def plot_metrics(data):
    """Plot images based on key and plot type"""
    redis_key, plot_type = data['key'], data['type']
    redis_type_2_plot_type = {
        'list': ['bar', 'dot'],
        'hash': ['line', 'pie'],
    }
   
    redis_type = redis_utils.redis_type(redis_key)
    if redis_type not in redis_type_2_plot_type:
        logging.error('do not support display for given redis type: {}'.format(redis_type))
        return
  
    if not plot_type:
        plot_type = redis_type_2_plot_type[redis_type][0]
    elif plot_type not in redis_type_2_plot_type[redis_type]:
        logging.error('do not support display for given plot type: {}'.format(plot_type))
        return
   
    width, height = 10, 5.5
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(width, height)

    values = (redis_utils.redis_range(redis_key)
              if value_type == 'list' else redis_utils.redis_get_dict_values(redis_key))

    if plot_type == 'bar':
       plot_bar(axs, values)
    elif plot_type == 'dot':
       plot_dot(axs, values)
    elif plot_type == 'pie':
       plot_pie(plt, axs, values)
    elif plot_type == 'line':
       plot_line(axs, redis_key, values)

    plt.tight_layout()
    html = mpld3.fig_to_html(fig)
    plt.close('all')
    return html


def plot_bar(axs, values):
    """Plot bar"""
    counters = Counter([round(float(x), 0) for x in values])
    bar_keys = counters.keys()
    bar_values = [counters[x] for x in bar_keys]
    axs.set_title('Distribution')
    axs.set_xlabel('Values')
    axs.set_ylabel('Frequency')
    axs.bar(bar_keys, bar_values)


def plot_line(axs, redis_key, values):
    """Plot line"""
    x_values = sorted(values.keys())
    y_values = [round(float(values[x]), 1) for x in x_values]
    axs.set_title('History Comparison')
    axs.set_xlabel('Dates')
    axs.set_ylabel('Values')
    axs.set_xticks(range(len(x_values)))
    axs.set_xticklabels(x_values)
    axs.plot(x_values, y_values, '-', label=redis_key)
    axs.legend()
    axs.grid(True)


def plot_pie(plt, axs, values):
    """Plot pie"""
    labels = values.keys()
    sizes = [round(float(values[x]), 1) for x in x_values] 
    explode = [0, 0, 0, 0]
    axs.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')


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
    'plot_metrics': plot_metrics,
}
