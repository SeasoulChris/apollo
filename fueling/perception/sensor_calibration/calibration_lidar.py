#!/usr/bin/env python
"""
This is a module to run perception benchmark on lidar data
"""
import operator
import os
import sys
import time
from datetime import datetime

import cv2
# import pyspark_utils.helper as spark_helper

if sys.version_info[0] >= 3:
    from cyber_py3.record import RecordReader, RecordWriter
else:
    from cyber_py.record import RecordReader, RecordWriter

from modules.drivers.proto.sensor_image_pb2 import CompressedImage

from fueling.common.base_pipeline import BasePipeline
# import fueling.common.bos_client as bos_client
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.common.redis_utils as redis_utils
import fueling.common.storage.bos_client as bos_client
import fueling.streaming.streaming_utils as streaming_utils

def execute_task(message_meta):
    """example task executing"""
    task_name, source_dir = message_meta
    """Execute task by task"""
    logging.info('executing task: {} with src_dir: {}'.format(task_name, source_dir))

    # Invoke benchmark binary
    logging.info('start to execute sensor calbiration service')

    data_time = datetime.now().strftime("%m-%d-%Y.txt")
    file_utils.makedirs(os.path.join(source_dir, 'outputs'))
    executable_dir = '/apollo/modules/data/fuel/fueling/perception/sensor_calibration/executable_bin/'
    if task_name == "Lidar_to_Gnss":
        executable_bin = executable_dir + "multi_lidar_to_gnss/multi_lidar_gnss_calibrator"
    else:
        logging.info('not support {} yet'.format(task_name))
        return
    """add lib path"""
    new_lib = '/apollo/modules/data/fuel/fueling/perception/sensor_calibration/executable_bin/multi_lidar_to_gnss/'
    if not new_lib in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ':'+new_lib
    """set command and config file example"""
    config_file = '/apollo/modules/data/fuel/fueling/perception/sensor_calibration/config_example/udelv_multi_lidar_gnss_calibrator_config.yaml'
    command = '{} --config {}'.format(executable_bin, config_file)
    logging.info('sensor calibration executable command is {}'.format(command))
    return_code = os.system(command)
    logging.info('return code for sensor calibration command is {}'.format(return_code))
    if return_code != 0:
        logging.error('failed to execute sensor caliration task for {}'.format(task_name))
        return
    logging.info('done executing')
class SensorCalibrationPipeline(BasePipeline):
    """Apply sensor calbiration to smartly extracted sensor frames"""

    # def __init__(self):
    #     """Initialize"""
    #     BasePipeline.__init__(self, 'sensor calibration')

    def run_test(self):
        """local mini test"""
        root_dir = '/apollo/data/extraced_data'
        original_path = '{}/Camera_Lidar_Calibration-2019-10-24-12-01'.format(root_dir)
        task_name = "Lidar_to_Gnss"
        self.run(original_path, task_name)
        logging.info("Sensor Calibration for .{} Task on {}: All Done".format(task_name, original_path))

    def run_prod(self, task_name = "Lidar_to_Gnss"):
        """Run Prod. production version"""        
        root_dir = bos_client.BOS_MOUNT_PATH
        original_path = '{}/modules/tools/sensor_calibration/data'.format(root_dir)        
        self.run([(original_path, task_name)])
        logging.info("Sensor Calibration for .{} Task on {}: All Done".format(task_name, original_path))

    def run(self, original_path, task_name):
        """Run the pipeline with given parameters"""
        self.to_rdd([(task_name, original_path)]).foreach(execute_task)

if __name__ == '__main__':
    SensorCalibrationPipeline().main()
