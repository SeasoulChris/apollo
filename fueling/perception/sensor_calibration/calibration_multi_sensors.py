#!/usr/bin/env python
"""This is a module to run perception benchmark on lidar data"""

from datetime import datetime
import glob
import os
import shutil

from fueling.common.base_pipeline import BasePipeline
from fueling.common.partners import partners
from fueling.perception.sensor_calibration.calibration_config import CalibrationConfig
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


def execute_task(message_meta):
    """example task executing, return results dir."""
    source_dir, output_dir = message_meta
    # from input config file, generating final fuel-using config file
    in_config_file = os.path.join(source_dir, 'sample_config.yaml')
    calib_config = CalibrationConfig()

    config_file = calib_config.generate_task_config_yaml(root_path=source_dir,
                                                         output_path=output_dir,
                                                         source_config_file=in_config_file)
    task_name = calib_config.get_task_name()
    """Execute task by task"""
    logging.info('type of {} is {}'.format(task_name, type(task_name)))
    logging.info('executing task: {} with src_dir: {}'.format(task_name, source_dir))
    # Invoke benchmark binary
    logging.info('start to execute sensor calbiration service')
    # Add lib path
    executable_dir = os.path.join(os.path.dirname(__file__), 'executable_bin')
    if executable_dir not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] = executable_dir + ':' + os.environ['LD_LIBRARY_PATH']
    os.system("echo $LD_LIBRARY_PATH")
    if task_name == 'lidar_to_gnss':
        executable_bin = os.path.join(executable_dir, 'multi_lidar_gnss_calibrator')
    elif task_name == 'camera_to_lidar':
        executable_bin = os.path.join(executable_dir, 'multi_grid_lidar_camera_calibrator')
        #logging.info('executable not ready, stay for tune')
        #return None
    else:
        logging.error('not support {} yet'.format(task_name))
        return None

    # set command
    command = f'{executable_bin} --config {config_file}'
    logging.info('sensor calibration executable command is {}'.format(command))

    return_code = os.system(command)
    if return_code == 0:
        logging.info('Finished sensor caliration.')
    else:
        logging.error('Failed to run sensor caliration for {}: {}'.format(task_name, return_code))
    return os.path.join(output_dir, 'results')


class SensorCalibrationPipeline(BasePipeline):
    """Apply sensor calbiration to smartly extracted sensor frames"""

    def _get_subdirs(self, d):
        """list add 1st-level task data directories under the root directory
        ignore hidden folders"""
        return [f for f in os.listdir(d) if not f.startswith('.') and
                os.path.isdir(os.path.join(d, f))]

    def run_test(self):
        """local mini test"""
        self.run('testdata/perception/sensor_calibration')

    def run_prod(self):
        """Run Prod. production version"""
        result_files = []
        try:
            result_files = self.run(self.FLAGS.get('input_data_path'))
        except:
            pass

        # Send result to job owner.
        receivers = email_utils.PERCEPTION_TEAM + email_utils.DATA_TEAM
        partner = partners.get(self.FLAGS['job_owner'])
        if partner:
            receivers.append(partner.email)

        if result_files:
            title = 'Your sensor calibration job is done!'
            content = {'Job Owner': self.FLAGS['job_owner'], 'Job ID': self.FLAGS['job_id']}
            email_utils.send_email_info(title, content, receivers, result_files)
        else:
            title = 'Your sensor calibration job failed!'
            content = (f'We are sorry. Please report the job id {self.FLAGS["job_id"]} to us at '
                       'IDG-apollo@baidu.com, so we can investigate.')
            email_utils.send_email_error(title, content, receivers)

    def run(self, job_dir):
        # If it's a partner job, move origin data to our storage before processing.
        if self.is_partner_job():
            job_dir = self.partner_storage().abs_path(job_dir)
            job_output_dir = self.our_storage().abs_path(
                os.path.join('modules/perception/sensor_calibration',
                             self.FLAGS['job_owner'], self.FLAGS['job_id']))
            file_utils.makedirs(job_output_dir)
            # TODO: Quick check on partner data.

        else:
            job_dir = self.our_storage().abs_path(job_dir)
            job_output_dir = job_dir

        subjobs = self._get_subdirs(job_dir)
        message_meta = [(os.path.join(job_dir, j), os.path.join(job_output_dir, j))
                        for j in subjobs]

        # Run the pipeline with given parameters.
        result_dirs = self.to_rdd(message_meta).map(execute_task).collect()

        result_files = []
        for result_dir in result_dirs:
            if result_dir:
                result_files.extend(glob.glob(os.path.join(result_dir, '*.yaml')))
                # camera to lidar calibration visualization results
                output_dir = glob.glob(os.path.join(result_dir, '*', 'out'))
                if os.path.isdir(output_dir[0]):
                    result_files.extend(glob.glob(os.path.join(output_dir[0], '*.jpg')))
                # lidar to gnss calibration visualization results.
                output_dir = os.path.join(result_dir, 'tmp', 'lidar')
                if os.path.isdir(output_dir):
                    result_files.extend(glob.glob(os.path.join(output_dir, '*_rgb.pcd')))
        logging.info(f"Sensor Calibration on data {job_dir}: All Done")
        logging.info(f"Generated {len(result_files)} results: {result_files}")
        return result_files


if __name__ == '__main__':
    SensorCalibrationPipeline().main()
