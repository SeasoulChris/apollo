#!/usr/bin/env python
"""This is a module to run perception benchmark on lidar data"""

from datetime import datetime
import os

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client
from fueling.perception.sensor_calibration.calibration_config import CalibrationConfig

def execute_task(message_meta):
    """example task executing"""
    source_dir = message_meta
    # from input config file, generating final fuel-using config file
    in_config_file = os.path.join(source_dir, 'sample_config.yaml')
    calib_config = CalibrationConfig()
    config_file = calib_config.generate_task_config_yaml(
                    root_path=source_dir,
                    source_config_file=in_config_file)
    task_name = calib_config.get_task_name()
    """Execute task by task"""
    logging.info('type of {} is {}'.format(task_name, type(task_name)))
    logging.info('executing task: {} with src_dir: {}'.format(task_name, source_dir))
    # Invoke benchmark binary
    logging.info('start to execute sensor calbiration service')

    executable_dir =  os.path.join(os.path.dirname(__file__), 'executable_bin')
    if task_name == 'lidar_to_gnss':
        executable_bin = os.path.join(executable_dir, 'multi_lidar_to_gnss',
                                    'multi_lidar_gnss_calibrator')
        # Add lib path
        new_lib = os.path.join(executable_dir, 'multi_lidar_to_gnss')
        if  not new_lib in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] = new_lib+':'+ os.environ['LD_LIBRARY_PATH']
        os.system("echo $LD_LIBRARY_PATH")
    elif task_name == 'camera_to_lidar':
        logging.info('executable not ready, stay for tune')
        return
    else:
        logging.error('not support {} yet'.format(task_name))
        return

    # set command
    command = f'{executable_bin} --config {config_file}'
    logging.info('sensor calibration executable command is {}'.format(command))

    return_code = os.system(command)
    if return_code == 0:
        logging.info('Finished sensor caliration.')
    else:
        logging.error('Failed to run sensor caliration for {}: {}'.format(task_name, return_code))


class SensorCalibrationPipeline(BasePipeline):
    """Apply sensor calbiration to smartly extracted sensor frames"""
    def _get_subdirs(self, d):
        """list add 1st-level task data directories under the root directory
        ignore hidden folders"""
        return list(filter(os.path.isdir,
            [os.path.join(d,f) for f in os.listdir(d) if not f.startswith('.')]))

    def run_test(self):
        """local mini test"""
        # root_dir = '/apollo/modules/data/fuel/testdata/perception/sensor_calibration/camera_to_lidar'
        # self.run(root_dir)
        self.run('testdata/perception/sensor_calibration')

    def run_prod(self):
        """Run Prod. production version"""
        self.run(self.FLAGS.get('input_data_path'))

    def run(self, root_dir):
        storage = self.partner_storage() or self.our_storage()
        root_dir = storage.abs_path(root_dir)
        original_paths = self._get_subdirs(root_dir)
        """Run the pipeline with given parameters"""
        self.to_rdd(original_paths).foreach(execute_task)
        logging.info("Sensor Calibration on data {}: All Done".format(original_paths))


if __name__ == '__main__':
    SensorCalibrationPipeline().main()
