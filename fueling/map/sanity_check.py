#!/usr/bin/env python
import cgi
import os


from cyber.python.cyber_py3.record import RecordReader

from fueling.common.job_utils import JobUtils
import fueling.common.context_utils as context_utils
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

# could be a list
CHANNELS = {record_utils.LOCALIZATION_CHANNEL}


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


def is_oversize_file(path):
    if file_utils.getInputDirDataSize(path) >= 5 * 1024 * 1024 * 1024:
        logging.error('The record file is oversize!')
        return True
    return False


def parse_error(input_data_path, output_data_path):
    if input_data_path == output_data_path:
        logging.error('The input data path must be different from the output data path!')
        return True


def missing_message_data(path, channels=CHANNELS):
    for record in list_records(path):
        logging.info("reading records %s" % record)
        reader = RecordReader(record)
        for channel in channels:
            logging.info("has %d messages" % reader.get_messagenumber(channel))
            if reader.get_messagenumber(channel) == 0:
                return True
    return False


def sanity_check(input_folder, output_folder, job_owner, job_id, email_receivers=None):
    err_msg = None
    if not len(list_records(input_folder)):
        err_msg = "One or more files are missing in %s" % input_folder
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_failure_code('E301')
    elif is_oversize_file(input_folder):
        err_msg = "The record file is oversize!"
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_failure_code('E300')
    elif parse_error(input_folder, output_folder):
        err_msg = "The input data path must be different from the output data path!"
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_failure_code('E307')
    elif missing_message_data(input_folder):
        err_msg = "Messages are missing in records of %s" % input_folder
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_failure_code('E302')
    else:
        logging.info("%s Passed sanity check." % input_folder)
        if context_utils.is_cloud() and email_receivers:
            title = 'Virtual-lane-generation data sanity check passed for {}'.format(job_owner)
            content = 'job_id={} input_folder={} output_folder={}\n' \
                'We are processing your job now. Please expect another email with results.'.format(
                    job_id, input_folder, output_folder)
            email_utils.send_email_info(title, content, email_receivers)
        return True

    if context_utils.is_cloud() and email_receivers:
        title = 'Error occurred during Virtual-lane-generation data sanity check for {}'.format(
            job_owner)
        content = 'job_id={} input_folder={}\n{}'.format(job_id, input_folder, cgi.escape(err_msg))
        email_utils.send_email_error(title, content, email_receivers)

    logging.error(err_msg)
    return False
