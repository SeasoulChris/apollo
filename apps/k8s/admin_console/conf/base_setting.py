#!/usr/bin/env python3
"""
Basic configuration module
"""

import os


class Config(object):
    """
    Basic configuration class
    """
    # Flask config
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.urandom(24)

    # Job config
    JOB_TYPE = {"A": "All", "VC": "vehicle_calibration",
                "SC": "sensor_calibration", "CP": "virtual_lane_generation"}
    SHOW_JOB_TYPE = {"A": "所有", "VC": "Vehicle Calibration", "SC": "Sensor Calibration",
                     "CP": "Virtual Lane Generation"}
    TIME_FIELD = {"All": 0, "7d": 7, "30d": 30, "1y": 365}
    SHOW_TIME_FIELD = {"All": "所有", "7d": "过去7天", "30d": "过去30天", "1y": "1年前"}
    AGGREGATED_FIELD = {"W": "周", "M": "月", "Y": "年"}
    AGGREGATED_BY = {"W": "week", "M": "month", "Y": "year"}
    BLACK_LIST = ["CH0000000"]
    FAILURE_CAUSE = {"E300": "数据包超过大小限制！请按照使用文档要求上传小于5G的数据包。",
                     "E301": "找不到所需要的输入文件！请按照使用文档检查输入文件的目录结构是否正确以及相应文件是否齐全。",
                     "E302": "数据包中缺少/apollo/localization/pose 消息！请按照使用文档要求录制正确的数据包。",
                     "E303": "输入文件中缺少激光雷达外参文件velodyne16_novatel_extrinsics_example.yaml，"
                             "无法生成定位地图。请按照使用文档提供相关文件。",
                     "E304": "数据包中缺少/apollo/sensor/gnss/odometry或者/apollo/sensor/gnss/ins_stat"
                             "或者/apollo/sensor/lidar16/compensator/PointCloud2消息，无法生成定位地图。"
                             "请按照使用文档要求录制正确的数据包。",
                     "E305": "服务超时，无法生成定位地图。请联系售后技术支持。",
                     "E306": "定位地图生成出错，请联系售后技术支持。",
                     "E200": "数据包超过大小限制！请按照使用文档要求上传小于1G的数据包。",
                     "E201": "找不到所需要的输入文件！请按照使用文档检查输入文件的目录结构是否正确以及相应文件是否齐全。",
                     "E202": "输入文件中缺少XXX配置文件。请按照使用文档提供相关文件。",
                     "E203": "配置文件XXX中缺少XXX配置项。请按照使用文档提供正确的配置文件",
                     "E204": "传感器标定出错，请联系售后技术支持。"}
