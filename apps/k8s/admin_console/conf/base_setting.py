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
                "SC": "sensor_calibration", "VLG": "virtual_lane_generation",
                "CP": "control_profiling", "PLMT": "perception_lidar_model_training",
                "OSPP": "open_space_planner_profiling", "PMT": "prediction_model_training",
                "CAT": "control_auto_tuning", "DM": "dynamic_model"}
    SHOW_JOB_TYPE = {"A": "所有", "VC": "Vehicle Calibration", "SC": "Sensor Calibration",
                     "VLG": "Virtual Lane Generation", "CP": "Control Profiling",
                     "PLMT": "Perception Lidar Model Training",
                     "OSPP": "Open Space Planner Profiling", "PMT": "Prediction Model Training",
                     "CAT": "Control Auto Tuning", "DM": "Dynamic Model"}
    TIME_FIELD = {"All": 0, "7d": 7, "30d": 30, "1y": 365}
    SHOW_TIME_FIELD = {"All": "所有", "7d": "过去7天", "30d": "过去30天", "1y": "1年前"}
    AGGREGATED_FIELD = {"W": "周", "M": "月", "Y": "年"}
    AGGREGATED_BY = {"W": "week", "M": "month", "Y": "year"}
    BLACK_LIST = ["CH0000000", "CH0000001", "CH0000002", "CH0000003", "None"]
    FAILURE_CAUSE = {"E100": "配置文件无法解析，配置文件解析有误。",
                     "E101": "查配置文件缺少相关参数: XXX。",
                     "E102": "查配置文件参数错误：XXX。",
                     "E103": "找不到所需要的输入文件！请按照使用文档检查输入文件的目录结构是否正确以及相应文件是否齐全。 ",
                     "E300": "数据包超过大小限制！请按照使用文档要求上传小于5G的数据包。",
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
                     "E204": "传感器标定出错，请联系售后技术支持。",
                     "E307": "输入路径和输出路径相同，请传入不同路径。",
                     "E400": "请检查提交任务的目录名称是否输入正确，或输入目录内是否缺少xxx文件。"
                             "例如：xxx可以是vehicle_param.pd.txt，或者是某个具体的record文件缺失或者找不到。",
                     "E401": "配置文件vehicle_param.pd.txt无法解析，请检查vehicle_param.pd.txt是否是车型配置目录下的文件。",
                     "E402": "请检查vehicle_param.pd.txt是否缺少以下参数和相应数据：vehicle_id、brake_deadzone、"
                             "throttle_deadzone、max_acceleration、max_deceleration。",
                     "E403": "xxxx record文件内缺失重要channel信息(chassis or localization channel), "
                             "请重新录制数据包",
                     "E500": "请检查提交任务的目录名称是否输入正确，或者输入目录内是否缺少xxx文件。"
                             "例如：xxx可以是vehicle_param.pd.txt，或者是某个具体的record文件缺失或者找不到。",
                     "E501": "配置文件vehicle_param.pd.txt无法解析，配置文件解析有误。",
                     "E502": "请检查vehicle_param.pd.txt是否缺少以下参数和相应数据：vehicle_id、wheel_base。",
                     "E503": "xxx record 文件内缺失重要channel信息：chassis、localization、control，请重新录制数据包。"
                     }
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
    SUPER_USER_NAME = "apollo"
    ROLE_TYPE = {
        "all": [],
        "system_admin": ["/jobs", "/reset_pwd", "/submit_job", "/statistics",
                         "/update_status", "/edit_quota", "/accounts", "/"],
        "service_specialist": ["/index", "/jobs"]
    }
    WHITE_URL = ["/static", "/login", "/logout", "/api"]
    ACCOUNT_SHOW_ACTION = {
        "Pending": ("Enable", "Reject"),
        "Rejected": (),
        "Enabled": ("Edit", "Disable"),
        "Disabled": ("Enable",),
        "Over-quota": ("Edit",),
        "Expired": ("Edit",)
    }
    SHOW_SERVICE_PACKAGE = {
        "TO": "试用包",
        "OY": "一年包"
    }
    ACCOUNT_SERVICE_QUOTA = {
        "TO": 15,
        "OY": 50
    }
    ACCOUNT_SERVICE_DAYS = {
        "TO": 30,
        "OY": 365
    }
    ACCOUNT_STATUS_FIELD = {
        "Enabled": "启用",
        "Pending": "待审批",
        "Rejected": "驳回",
        "Disabled": "停用",
        "Expired": "过期",
        "Over-quota": "超额"
    }
    ACCOUNT_REGION_FIELD = {
        "bj": "北京",
        "su": "苏州",
        "gz": "广州"
    }
    ACCOUNT_ACTION_FIELD = {
        "Enable": "启用",
        "Reject": "驳回",
        "Edit": "编辑",
        "Disable": "停用"
    }

    ADMIN_AUTH_DIGEST = {
        "admin_console": "admin_console:digest"
    }
