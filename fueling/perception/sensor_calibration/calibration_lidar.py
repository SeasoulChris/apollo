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
    source_dir, task_name = message_meta
    """Execute task by task"""
    logging.info('executing task: {} with src_dir: {}'.format(task_name, source_dir))
    # Invoke benchmark binary
    logging.info('start to execute sensor calbiration service')

    file_utils.makedirs(os.path.join(source_dir, 'outputs'))
    executable_dir =  os.path.join(os.path.dirname(__file__), 'executable_bin')
    if task_name == 'lidar_to_gnss':
        executable_bin = os.path.join(executable_dir, 'multi_lidar_to_gnss',
                                    'multi_lidar_gnss_calibrator')
        # Add lib path
        new_lib = os.path.join(executable_dir, 'multi_lidar_to_gnss')
        if  not new_lib in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] = new_lib+':'+ os.environ['LD_LIBRARY_PATH']
        os.system("echo $LD_LIBRARY_PATH")
    else:
        logging.error('not support {} yet'.format(task_name))
        return

    # set command and config file example
    config_file = os.path.join(source_dir, 'config.yaml')
    command = f'{executable_bin} --config {config_file}'
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
        task_name = 'lidar_to_gnss'
        root_dir = '/apollo/modules/data/fuel/testdata/perception/sensor_calibration'
        original_path = os.path.join(root_dir, task_name)
        self.run(original_path, task_name)

    def run_prod(self, task_name='lidar_to_gnss'):
        """Run Prod. production version"""
        root_dir = bos_client.BOS_MOUNT_PATH
        original_path = os.path.join(root_dir, 'modules/tools/sensor_calibration/data')
        self.run(original_path, task_name)

    def run(self, original_path, task_name):
        """Run the pipeline with given parameters"""
        self.to_rdd([(original_path, task_name)]).foreach(execute_task)
        logging.info("Sensor Calibration for {} on {}: All Done".format(task_name, original_path))


if __name__ == '__main__':
    SensorCalibrationPipeline().main()
