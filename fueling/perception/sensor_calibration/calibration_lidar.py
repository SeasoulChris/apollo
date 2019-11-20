#!/usr/bin/env python
"""This is a module to run perception benchmark on lidar data"""

import os
from datetime import datetime

import cv2

from modules.drivers.proto.sensor_image_pb2 import CompressedImage

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client


def execute_task(message_meta):
    """example task executing"""
    task_name, source_dir = message_meta
    """Execute task by task"""
    logging.info('executing task: {} with src_dir: {}'.format(task_name, source_dir))

    # Invoke benchmark binary
    logging.info('start to execute sensor calbiration service')

    file_utils.makedirs(os.path.join(source_dir, 'outputs'))
    executable_dir = '/apollo/modules/data/fuel/fueling/perception/sensor_calibration/executable_bin/'
    if task_name == "Lidar_to_Gnss":
        executable_bin = executable_dir + "multi_lidar_to_gnss/multi_lidar_gnss_calibrator"
    else:
        logging.error('not support {} yet'.format(task_name))
        return

    # Add lib path
    new_lib = '/apollo/modules/data/fuel/fueling/perception/sensor_calibration/executable_bin/multi_lidar_to_gnss/'
    if not new_lib in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ':' + new_lib

    # set command and config file example
    config_file = ('/apollo/modules/data/fuel/fueling/perception/sensor_calibration/'
                   'config_example/udelv_multi_lidar_gnss_calibrator_config.yaml')
    command = '{} --config {}'.format(executable_bin, config_file)
    logging.info('sensor calibration executable command is {}'.format(command))

    return_code = os.system(command)
    if return_code == 0:
        logging.info('Finished sensor caliration.')
    else:
        logging.error('Failed to run sensor caliration for {}: {}'.format(task_name, return_code))


class SensorCalibrationPipeline(BasePipeline):
    """Apply sensor calbiration to smartly extracted sensor frames"""

    def run_test(self):
        """local mini test"""
        root_dir = '/apollo/data/extraced_data'
        original_path =  os.path.join(root_dir, 'Camera_Lidar_Calibration-2019-10-24-12-01')
        task_name = "Lidar_to_Gnss"
        self.run(original_path, task_name)

    def run_prod(self, task_name="Lidar_to_Gnss"):
        """Run Prod. production version"""
        root_dir = bos_client.BOS_MOUNT_PATH
        original_path = os.path.join(root_dir, 'modules/tools/sensor_calibration/data')
        self.run(original_path, task_name)

    def run(self, original_path, task_name):
        """Run the pipeline with given parameters"""
        self.to_rdd([(task_name, original_path)]).foreach(execute_task)
        logging.info("Sensor Calibration for {} on {}: All Done".format(task_name, original_path))


if __name__ == '__main__':
    SensorCalibrationPipeline().main()
