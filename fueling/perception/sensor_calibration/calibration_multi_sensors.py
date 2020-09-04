#!/usr/bin/env python
"""This is a module to run perception benchmark on lidar data"""

from datetime import datetime
import glob
import os
import time

from fueling.common.base_pipeline import BasePipeline
from fueling.common.job_utils import JobUtils
from fueling.perception.sensor_calibration.sanity_check import sanity_check
from fueling.perception.sensor_calibration.calibration_config import CalibrationConfig
import fueling.common.context_utils as context_utils
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.redis_utils as redis_utils
import fueling.common.storage.bos_client as bos_client


class SensorCalibrationPipeline(BasePipeline):
    """Apply sensor calbiration to smartly extracted sensor frames"""

    def _get_subdirs(self, d):
        """list add 1st-level task data directories under the root directory
        ignore hidden folders"""
        return [f for f in os.listdir(d) if not f.startswith('.')
                and os.path.isdir(os.path.join(d, f))]

    def run_test(self):
        """local mini test"""
        self.is_on_cloud = context_utils.is_cloud()
        self.run_internal('/fuel/testdata/perception/sensor_calibration')

    def run(self):
        """Run Prod. production version"""
        sub_type = set()
        result_files = []
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        self.is_on_cloud = context_utils.is_cloud()
        src_prefix = self.FLAGS.get('input_data_path') or "test/sensor_calibration"
        object_storage = self.partner_storage() or self.our_storage()
        source_dir = object_storage.abs_path(src_prefix)

        job_type, job_size = 'SENSOR_CALIBRATION', file_utils.getDirSize(source_dir)
        redis_key = F'External_Partner_Job.{job_owner}.{job_type}.{job_id}'
        redis_value = {'begin_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                       'job_size': job_size,
                       'job_status': 'running'}
        if self.is_on_cloud:
            redis_utils.redis_extend_dict(redis_key, redis_value)
            JobUtils(job_id).save_job_input_data_size(source_dir)
            JobUtils(job_id).save_job_sub_type('')

        # Send result to job owner.
        receivers = email_utils.PERCEPTION_TEAM + email_utils.DATA_TEAM + email_utils.D_KIT_TEAM
        if os.environ.get('PARTNER_EMAIL'):
            receivers.append(os.environ.get('PARTNER_EMAIL'))

        if not sanity_check(source_dir, job_owner, job_id, receivers):
            raise Exception("Sanity_check failed!")

        self.error_text = 'Calibration error, please contact after-sales technical support'

        try:
            result = self.run_internal(src_prefix)
        except BaseException as e:
            if self.is_on_cloud:
                JobUtils(job_id).save_job_failure_code('E209')
                JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                     self.error_text, False)
            logging.error(e)

        logging.info(f"Generated sub_type {len(sub_type)} results: {sub_type}")
        sub_job_type = 'All'

        if result:
            result_files, sub_type = result
            if len(sub_type) == 1:
                sub_job_type = sub_type.pop()

            title = 'Your sensor calibration job is done!'
            content = {'Job Owner': job_owner, 'Job ID': job_id}
            if self.is_on_cloud:
                email_utils.send_email_info(title, content, receivers, result_files)
                redis_value = {'end_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                               'job_status': 'success', 'sub_type': sub_job_type}
                redis_utils.redis_extend_dict(redis_key, redis_value)
        else:
            title = 'Your sensor calibration job failed!'
            content = (f'We are sorry. Please report the job id {self.FLAGS["job_id"]} to us at '
                       'IDG-apollo@baidu.com, so we can investigate.')
            if self.is_on_cloud:
                email_utils.send_email_error(title, content, receivers)
                redis_value = {'end_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                               'job_status': 'failed', 'sub_type': sub_job_type}
                redis_utils.redis_extend_dict(redis_key, redis_value)
            logging.error('Failed to process sensor calibration job')
            raise Exception("Failed to process sensor calibration job!")
        if self.is_on_cloud:
            job_result = JobUtils(job_id).get_job_info()
            for job_info in job_result:
                if (int(time.mktime(datetime.now().timetuple())
                        - time.mktime(job_info['start_time'].timetuple()))) > 259200:
                    JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                         self.error_text, False)
                    JobUtils(job_id).save_job_failure_code('E209')
                    raise Exception("The running time is more than three days!")
            JobUtils(job_id).save_job_sub_type(sub_job_type)
            JobUtils(job_id).save_job_progress(100)

    def run_internal(self, job_dir):
        # If it's a partner job, move origin data to our storage before processing.
        if self.is_partner_job():
            job_dir = self.partner_storage().abs_path(job_dir)
            job_id = self.FLAGS.get('job_id')
            dst_prefix = self.FLAGS.get('output_data_path') or job_dir
            origin_prefix = os.path.join(dst_prefix, job_id)
            job_output_dir = self.partner_storage().abs_path(origin_prefix)
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
        for temp_dir in result_dirs:
            if temp_dir:
                result_dir, task_name = temp_dir
                if result_dir:
                    task_types.add(task_name)
                    result_files.extend(glob.glob(os.path.join(result_dir, '*.yaml')))
            else:
                logging.error(f"result_dirs: {result_dirs}: has no results")
                return None
        logging.info(f"Sensor Calibration on data {job_dir}: All Done")
        logging.info(f"Generated {len(result_files)} results: {result_files}")
        logging.info(f"Generated task_types {len(task_types)} results: {task_types}")
        return result_files, task_types

    def execute_task(self, message_meta):
        """example task executing, return results dir."""
        logging.info('start to execute sensor calbiration service')
        job_id = self.FLAGS.get('job_id')
        self.index += 1

        if self.is_on_cloud:
            JobUtils(job_id).save_job_progress(10 + (80 // self.num) * (self.index - 1))
        source_dir, output_dir, executable_dir = message_meta

        logging.info(F'executable dir: {executable_dir}, exist? {os.path.exists(executable_dir)}')

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
        if executable_dir not in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] = executable_dir + ':' + os.environ['LD_LIBRARY_PATH']
        os.system("echo $LD_LIBRARY_PATH")

        if task_name == 'lidar_to_gnss':
            executable_bin = os.path.join(executable_dir, 'multi_lidar_gnss_calibrator')
        elif task_name == 'camera_to_lidar':
            executable_bin = os.path.join(executable_dir, 'multi_grid_lidar_camera_calibrator')
        else:
            logging.error('not support {} yet'.format(task_name))
            return None

        if not os.path.exists(executable_bin):
            if self.is_on_cloud:
                JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                     self.error_text, False)
                JobUtils(job_id).save_job_failure_code('E209')
            logging.error('{} is not exists!'.format(executable_bin))
            return None
        # set command
        command = f'{executable_bin} --config {config_file}'
        logging.info('sensor calibration executable command is {}'.format(command))

        return_code = os.system(command)
        if return_code == 0:
            logging.info('Finished sensor caliration.')
        else:
            if self.is_on_cloud:
                if return_code == 10:
                    JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                         self.error_text, False)
                    JobUtils(job_id).save_job_failure_code('E206')
                elif return_code == 11:
                    JobUtils(job_id).save_job_failure_code('E207')
                elif return_code == 12:
                    JobUtils(job_id).save_job_failure_code('E208')
                else:
                    JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                         self.error_text, False)
                    JobUtils(job_id).save_job_failure_code('E209')
            logging.error('Failed to run sensor caliration for {}: {}'.format(task_name,
                                                                              return_code))
            time.sleep(60 * 3)
            return None
        if self.is_on_cloud:
            JobUtils(job_id).save_job_progress(10 + (80 // self.num) * self.index)
        return os.path.join(output_dir, 'results'), task_name


if __name__ == '__main__':
    SensorCalibrationPipeline().main()
