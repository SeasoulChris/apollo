""" Control feature extraction related utils. """
#!/usr/bin/env python

import numpy as np

import fueling.common.colored_glog as glog
import fueling.common.h5_utils as h5_utils
import fueling.common.record_utils as record_utils
import fueling.common.time_utils as time_utils


def compare_controller_type(controller_type, conf_controllers, CONTROL_CONF):
    """ Compare the selected controller type in records and the corrsponding control_conf.pb.txt """
    active_controllers = [];
    for controllers in conf_controllers:
        if controllers == CONTROL_CONF.LAT_CONTROLLER:
            active_controllers.append("Lat_Controller")
        elif controllers == CONTROL_CONF.LON_CONTROLLER:
            active_controllers.append("Lon_Controller")
        elif controllers == CONTROL_CONF.MPC_CONTROLLER:
            active_controllers.append("Mpc_Controller")

    if (len(active_controllers) == 1 and active_controllers.count('Mpc_Controller') == 1):
        if (controller_type == "Mpc_Controller"):
            return "Selected controller type matches the setting in the control_conf"
        else:
            return "Warning: Selected controller type does NOT match the control_conf"
    elif (len(active_controllers) == 2 and active_controllers.count('Lon_Controller') == 1 and
        active_controllers.count('Lat_Controller') == 1):
        if (controller_type == "Lon_Lat_Controller"):
            return "Selected controller type matches the setting in the control_conf"
        else:
            return active_controllers
    else:
        return "Warning: The controller setting in the control_conf has problem"


def get_vehicle_of_dirs(dir_to_records_rdd):
    """
    Extract HMIStatus.current_vehicle from each dir.
    Convert RDD(dir, record) to RDD(dir, vehicle).
    """
    def _get_vehicle_from_records(records):
        reader = record_utils.read_record([record_utils.HMI_STATUS_CHANNEL])
        for record in records:
            glog.info('Try getting vehicle name from {}'.format(record))
            for msg in reader(record):
                hmi_status = record_utils.message_to_proto(msg)
                vehicle = hmi_status.current_vehicle
                glog.info('Get vehicle name "{}" from record {}'.format(vehicle, record))
                return vehicle
        glog.info('Failed to get vehicle name')
        return ''
    return dir_to_records_rdd.groupByKey().mapValues(_get_vehicle_from_records)


def get_controller_of_dirs(dir_to_records_rdd):
    """
    Extract CONTROL_CHANNEL.active_controllers from each dir.
    Convert RDD(dir, record) to RDD(dir, controller).
    """
    def _get_controller_from_records(records):
        reader = record_utils.read_record([record_utils.CONTROL_CHANNEL])
        for record in records:
            glog.info('Try getting controller types from {}'.format(record))
            for msg in reader(record):
                controlcmd = record_utils.message_to_proto(msg)
                if (controlcmd.debug.simple_lon_debug is not '' and
                    controlcmd.debug.simple_lat_debug is not ''):
                    controller = 'Lon_Lat_Controller'
                elif (controlcmd.debug.simple_mpc_debug is not ''):
                    controller = 'Mpc_Controller'
                else:
                    controller = 'Abnormal_Controller'
                glog.info('Get control type "{}" from record {}'.format(controller, record))
                return controller
        glog.info('Failed to get controller type')
        return ''
    return dir_to_records_rdd.groupByKey().mapValues(_get_controller_from_records)


def gen_pre_segment(dir_to_msg):
    """Generate new key which contains a segment id part."""
    task_dir, msg = dir_to_msg
    time_stamp = time_utils.msg_time_to_datetime(msg.timestamp)
    segment_id = time_stamp.strftime('%Y%m%d-%H%M')
    return ((task_dir, segment_id), msg)


def extract_data(controller_type):
    "extract data set from msg based on the active controller types"
    def get_data_point(elem):
        """ extract data from msg """
        control = elem[1]
        data_array = []
        if (controller_type.count('Lon_Lat_Controller') == 1):
            control_lon = elem[1].debug.simple_lon_debug
            control_lat = elem[1].debug.simple_lat_debug
            data_array = np.array([
                # Features: "Refernce" category
                control_lon.station_reference,          # 0
                control_lon.speed_reference,            # 1
                control_lon.preview_acceleration_reference,  # 2
                control_lat.ref_heading,                # 3
                control_lat.curvature,                  # 4
                # Features: "Error" category
                control_lon.station_error,              # 5
                control_lon.speed_error,                # 6
                control_lat.lateral_error,              # 7
                control_lat.lateral_error_rate,         # 8
                control_lat.heading_error,              # 9
                control_lat.heading_error_rate,         # 10
                # Features: "Command" category
                control.throttle,                      # 11
                control.brake,                         # 12
                control.acceleration,                  # 13
                control.steering_target,               # 14
                # Features: "Status" category
                control_lat.ref_speed,                  # 15
                control_lat.heading,                    # 16
            ])
        elif (controller_type.count('Mpc_Controller') == 1):
            control_mpc = elem[1].debug.simple_mpc_debug
            data_array = np.array([
                # Features: "Refernce" category
                control_mpc.station_reference,           # 0
                control_mpc.speed_reference,             # 1
                control_mpc.acceleration_reference,      # 2
                control_mpc.ref_heading,                 # 3
                control_mpc.curvature,                   # 4
                # Features: "Error" category
                control_mpc.station_error,               # 5
                control_mpc.speed_error,                 # 6
                control_mpc.lateral_error,               # 7
                control_mpc.lateral_error_rate,          # 8
                control_mpc.heading_error,               # 9
                control_mpc.heading_error_rate,          # 10
                # Features: "Command" category
                control.throttle,                       # 11
                control.brake,                          # 12
                control.acceleration,                   # 13
                control.steering_target,                # 14
                # Features: "Status" category
                control_mpc.ref_speed,                   # 15
                control_mpc.heading,                     # 16
            ])
        else:
            return None
        return ((elem[0][0], control.header.timestamp_sec), data_array)
    return get_data_point


def to_list(elem):
    """convert element to list"""
    return [elem]

def append(orig_elem, app_elem):
    """append another element to the revious element"""
    orig_elem.append((app_elem))
    return orig_elem

def extend(orig_elem, app_elem):
    """extend the original list"""
    orig_elem.extend(app_elem)
    return orig_elem


def gen_segment_by_key(elem):
    """ generate segment by key feature """
    segments = elem[0]
    for i in range(1, len(elem)):
        segments = np.vstack([segments, elem[i]])
    return segments

def write_h5_with_key(elem, origin_prefix, target_prefix, vehicle_type):
    """write to h5 file, use vehicle type plus controller type as file name"""
    key_control = elem[0][1]
    folder_path = str(elem[0][0])
    folder_path = folder_path.replace(origin_prefix, target_prefix, 1)
    file_name = vehicle_type +'_' + key_control
    h5_utils.write_h5(elem[1], folder_path, file_name)
    return elem[0]
