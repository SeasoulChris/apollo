#!/usr/bin/env python
import cgi
import math
import os

import google.protobuf.text_format as text_format

from cyber.python.cyber_py3.record import RecordReader
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.common.job_utils import JobUtils
import fueling.common.email_utils as email_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils

# could be a list
ConfFile = 'vehicle_param.pb.txt'
CHANNELS = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}


def list_records(path):
    logging.info("in list_records:%s" % path)
    records = []
    for (dirpath, _, filenames) in os.walk(path):
        logging.info('filenames: %s' % filenames)
        logging.info('dirpath %s' % dirpath)
        for filename in filenames:
            end_file = os.path.join(dirpath, filename)
            logging.info("end_files: %s" % end_file)
            if record_utils.is_record_file(end_file):
                records.append(end_file)
    return records


def missing_file(path):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    logging.info("vehicles %s" % vehicles)
    result = []
    for vehicle in vehicles:
        # config file
        conf = os.path.join(path, vehicle, ConfFile)
        logging.info("vehicles conf %s" % conf)
        if os.path.exists(conf) is False:
            logging.error('Missing configuration file in %s' % vehicle)
            result.append(ConfFile)
        # record file
        logging.info("list of records:" % list_records(os.path.join(path, vehicle)))
        if len(list_records(os.path.join(path, vehicle))) == 0:
            logging.error('No record files in %s' % vehicle)
            result.append("record")
        if len(result):
            return True, result
    return False, []


def parse_error(path):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    pb_value = vehicle_config_pb2.VehicleConfig()
    for vehicle in vehicles:
        conf = os.path.join(path, vehicle, ConfFile)
        try:
            proto_utils.get_pb_from_text_file(conf, pb_value)
            return False
        except text_format.ParseError:
            logging.error('Error: Cannot parse %s as binary or text proto' % conf)
            return True


def check_vehicle_id(conf):
    # print(conf.HasField('vehicle_id.other_unique_id'))
    vehicle_id = conf.vehicle_param.vehicle_id
    if vehicle_id.vin or vehicle_id.plate or vehicle_id.other_unique_id:
        return True
    logging.error("Error: No vehicle ID")
    return False


def missing_field(path):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    logging.info("vehicles in missing field: %s" % vehicles)
    result = []
    for vehicle in vehicles:
        conf_file = os.path.join(path, vehicle, ConfFile)
        logging.info("conf_file: %s" % conf_file)
        # reset for each vehicle to avoid overwrited
        pb_value = vehicle_config_pb2.VehicleConfig()
        conf = proto_utils.get_pb_from_text_file(conf_file, pb_value)
        logging.info("vehicles conf %s" % conf)
        if not check_vehicle_id(conf):
            result.append("vehicle_id")
        # required field
        fields = [conf.vehicle_param.brake_deadzone,
                  conf.vehicle_param.throttle_deadzone,
                  conf.vehicle_param.max_acceleration,
                  conf.vehicle_param.max_deceleration]
        # for value in conf.vehicle_param:
        # has field is always true since a default value is given
        for field in fields:
            if math.isnan(field):
                result.append(field)
        if len(result):
            return True, result
    return False, result


def missing_message_data(path, channels=CHANNELS):
    result = []
    for record in list_records(path):
        logging.info("reading records %s" % record)
        reader = RecordReader(record)
        for channel in channels:
            logging.info("has %d messages" % reader.get_messagenumber(channel))
            if reader.get_messagenumber(channel) == 0:
                result.append(record)
    if len(result):
        return True, result
    return False, []


def sanity_check(input_folder, job_owner, job_id, task_name, email_receivers=None):
    err_msg = None
    field_flag, field_result = missing_field(input_folder)
    channel_flag, channel_result = missing_message_data(input_folder)
    file_flag, file_result = missing_file(input_folder)
    if file_flag:
        err_msg = "One or more files are missing in %s" % input_folder
        JobUtils(job_id).save_job_failure_code('E400')
        JobUtils(job_id).save_job_failure_detail(file_result)
    elif parse_error(input_folder):
        err_msg = "Config file cannot be parsed in %s" % input_folder
        JobUtils(job_id).save_job_failure_code('E401')
    elif field_flag:
        err_msg = "One or more fields are missing in config file %s" % input_folder
        JobUtils(job_id).save_job_failure_code('E402')
        JobUtils(job_id).save_job_failure_detail(field_result)
    elif channel_flag:
        err_msg = "Messages are missing in records of %s" % input_folder
        JobUtils(job_id).save_job_failure_code('E403')
        JobUtils(job_id).save_job_failure_detail(channel_result)
    else:
        logging.info("%s Passed sanity check." % input_folder)
        if email_receivers:
            title = F'{task_name} data sanity check passed for {job_owner}'
            content = 'job_id={} input_folder={}\n' \
                'We are processing your job now. Please expect another email with results.'.format(
                    job_id, input_folder)
            email_utils.send_email_info(title, content, email_receivers)
        return True

    if email_receivers:
        title = F'Error occurred during {task_name} data sanity check for {job_owner}'
        content = 'job_id={} input_folder={}\n{}'.format(job_id, input_folder, cgi.escape(err_msg))
        email_utils.send_email_error(title, content, email_receivers)

    logging.error(err_msg)
    return False
