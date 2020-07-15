#!/usr/bin/env python
"""This is a module to run perception benchmark on lidar data"""

from datetime import datetime
import glob
import os
import shutil
import time

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.common.job_utils import JobUtils
from fueling.common.partners import partners
from fueling.perception.sensor_calibration.calibration_config import CalibrationConfig
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.redis_utils as redis_utils
import fueling.common.storage.bos_client as bos_client

flags.DEFINE_string('vehicle_sn', None, ' verhicle_sn of parner users.')
flags.DEFINE_string('job_type', 'verhicle_calibration', 'job type.')


class SensorCalibrationPipeline(BasePipeline):
    """Apply sensor calbiration to smartly extracted sensor frames"""

    def _get_subdirs(self, d):
        """list add 1st-level task data directories under the root directory
        ignore hidden folders"""
        return [f for f in os.listdir(d) if not f.startswith('.')
                and os.path.isdir(os.path.join(d, f))]

    def run_test(self):
        """local mini test"""
        self.run_internal('testdata/perception/sensor_calibration')

    def run(self):
        """Run Prod. production version"""
        sub_type = set()
        result_files = []
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        object_storage = self.partner_storage() or self.our_storage()
        source_dir = object_storage.abs_path(self.FLAGS.get('input_data_path'))

        job_type, job_size = 'SENSOR_CALIBRATION', file_utils.getDirSize(source_dir)
        redis_key = F'External_Partner_Job.{job_owner}.{job_type}.{job_id}'
        redis_value = {'begin_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                       'job_size': job_size,
                       'job_status': 'running'}
        redis_utils.redis_extend_dict(redis_key, redis_value)
        JobUtils(job_id).save_job_input_data_size(source_dir)
        JobUtils(job_id).save_job_sub_type('')
        if file_utils.getInputDirDataSize(source_dir) >= 1 * 1024 * 1024 * 1024:
            JobUtils(job_id).save_job_failure_code('E200')
            return
        try:
            result_files, sub_type = self.run_internal(self.FLAGS.get('input_data_path'))
        except BaseException as e:
            JobUtils(job_id).save_job_failure_code('E204')
            JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                 'Calibration error, \
                                                 please contact after-sales technical support',
                                                 False)
            logging.error(e)

        # Send result to job owner.
        receivers = email_utils.PERCEPTION_TEAM + email_utils.DATA_TEAM + email_utils.D_KIT_TEAM
        partner = partners.get(job_owner)
        if partner:
            receivers.append(partner.email)

        logging.info(f"Generated sub_type {len(sub_type)} results: {sub_type}")
        sub_job_type = 'All'
        if len(sub_type) == 1:
            sub_job_type = sub_type.pop()

        if result_files:
            title = 'Your sensor calibration job is done!'
            content = {'Job Owner': job_owner, 'Job ID': job_id}
            email_utils.send_email_info(title, content, receivers, result_files)
            redis_value = {'end_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                           'job_status': 'success', 'sub_type': sub_job_type}
            redis_utils.redis_extend_dict(redis_key, redis_value)
        else:
            title = 'Your sensor calibration job failed!'
            content = (f'We are sorry. Please report the job id {self.FLAGS["job_id"]} to us at '
                       'IDG-apollo@baidu.com, so we can investigate.')
            email_utils.send_email_error(title, content, receivers)
            redis_value = {'end_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                           'job_status': 'failed', 'sub_type': sub_job_type}
            redis_utils.redis_extend_dict(redis_key, redis_value)
            logging.error('Failed to process sensor calibration job')
        result = JobUtils(job_id).get_job_info()
        for job_info in result:
            if (int(time.mktime(datetime.now().timetuple())
                    - time.mktime(job_info['start_time'].timetuple()))) > 259200:
                JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                     'Calibration error, \
                                                     please contact after-sales technical support',
                                                     False)
                JobUtils(job_id).save_job_failure_code('E204')
        JobUtils(job_id).save_job_sub_type(sub_job_type)
        JobUtils(job_id).save_job_progress(100)

    def run_internal(self, job_dir):
        # If it's a partner job, move origin data to our storage before processing.
        if self.is_partner_job():
            job_dir = self.partner_storage().abs_path(job_dir)

            job_output_dir = self.partner_storage().abs_path(self.FLAGS.get('output_data_path'))
            if not job_output_dir.startswith(bos_client.PARTNER_BOS_MOUNT_PATH):
                logging.error(F'Wrong job_output_dir {job_output_dir}')

            file_utils.makedirs(job_output_dir)
            # TODO: Quick check on partner data.
        else:
            job_dir = self.our_storage().abs_path(job_dir)
            job_output_dir = job_dir

        logging.info(F'job dir: {job_dir}')

        subjobs = self._get_subdirs(job_dir)
        logging.info(F'subjobs : {subjobs}')

        executable_dir = 'modules/perception/sensor_calibration/executable_bin'
        message_meta = [(os.path.join(job_dir, j),
                         os.path.join(job_output_dir, j),
                         self.our_storage().abs_path(executable_dir))
                        for j in subjobs]

        self.num = len(subjobs)
        self.index = 0

        # Run the pipeline with given parameters.
        result_dirs = self.to_rdd(message_meta).map(self.execute_task).collect()

        logging.info(f"result_dirs {result_dirs}: All Done")

        task_types = set()
        result_files = []
        for result_dir, task_name in result_dirs:
            if result_dir:
                task_types.add(task_name)
                result_files.extend(glob.glob(os.path.join(result_dir, '*.yaml')))
        logging.info(f"Sensor Calibration on data {job_dir}: All Done")
        logging.info(f"Generated {len(result_files)} results: {result_files}")
        logging.info(f"Generated task_types {len(task_types)} results: {task_types}")
        return result_files, task_types

    def execute_task(self, message_meta):
        """example task executing, return results dir."""
        logging.info('start to execute sensor calbiration service')
        job_id = self.FLAGS.get('job_id')
        self.index += 1

        JobUtils(job_id).save_job_progress(10 + (80 // self.num) * (self.index - 1))
        source_dir, output_dir, executable_dir = message_meta

        logging.info(F'executable dir: {executable_dir}, exist? {os.path.exists(executable_dir)}')

        # from input config file, generating final fuel-using config file
        in_config_file = os.path.join(source_dir, 'sample_config.yaml')
        if not os.path.exists(in_config_file):
            JobUtils(job_id).save_job_failure_code('E202')
        calib_config = CalibrationConfig()

        config_file = calib_config.generate_task_config_yaml(root_path=source_dir,
                                                             output_path=output_dir,
                                                             source_config_file=in_config_file)
        task_name = calib_config.get_task_name()
        """Execute task by task"""
        logging.info('type of {} is {}'.format(task_name, type(task_name)))
        logging.info('executing task: {} with src_dir: {}'.format(task_name, source_dir))

        # Invoke benchmark binary
        if executable_dir not in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] = executable_dir + ':' + os.environ['LD_LIBRARY_PATH']
        os.system("echo $LD_LIBRARY_PATH")

        if task_name == 'lidar_to_gnss':
            executable_bin = os.path.join(executable_dir, 'multi_lidar_gnss_calibrator')
        elif task_name == 'camera_to_lidar':
            executable_bin = os.path.join(executable_dir, 'multi_grid_lidar_camera_calibrator')
        else:
            JobUtils(job_id).save_job_failure_code('E203')
            logging.error('not support {} yet'.format(task_name))
            time.sleep(60 * 3)
            return None

        if not os.path.exists(executable_bin):
            JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                 'Calibration error, \
                                                 please contact after-sales technical support',
                                                 False)
            JobUtils(job_id).save_job_failure_code('E204')
        # set command
        command = f'{executable_bin} --config {config_file}'
        logging.info('sensor calibration executable command is {}'.format(command))

        return_code = os.system(command)
        if return_code == 0:
            logging.info('Finished sensor caliration.')
        else:
            JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                 'Calibration error, \
                                                 please contact after-sales technical support',
                                                 False)
            JobUtils(job_id).save_job_failure_code('E204')
            logging.error('Failed to run sensor caliration for {}: {}'.format(task_name,
                                                                              return_code))
            time.sleep(60 * 3)

        JobUtils(job_id).save_job_progress(10 + (80 // self.num) * self.index)
        return os.path.join(output_dir, 'results'), task_name


if __name__ == '__main__':
    SensorCalibrationPipeline().main()
