#!/usr/bin/env python
import cgi
import os

from fueling.common.job_utils import JobUtils
import fueling.common.context_utils as context_utils
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


def missing_config_file(path):
    sample_config_files = []
    for (dirpath, _, filenames) in os.walk(path):
        for filename in filenames:
            if filename == 'sample_config.yaml':
                end_file = os.path.join(dirpath, filename)
                sample_config_files.append(end_file)
    if (len(sample_config_files) == 0):
        return True, []
    return False, sample_config_files


def get_line_info(line):
    line_info = line.split(':', 1)[1]
    line_info = line_info.strip()
    return line_info


def missing_calibration_task(sample_config_files):
    for sample_config_file in sample_config_files:
        print(sample_config_file)
        flag = False
        try:
            file_object = open(sample_config_file, 'r')
            for line in file_object:
                if 'calibration_task' in line:
                    flag = True
                    break
        except Exception as e:
            logging.error(e)
            return False
        if not flag:
            return True
    return False


def get_calibration_task_info(sample_config_files):
    camera_lidar_task_list = ['data_path', 'extrinsic', 'intrinsic']
    lidar_gnss_task_list = ['odometry_file', 'sensor_files_directory']
    lidar_gnss_configs = []
    camera_lidar_configs = []
    for sample_config_file in sample_config_files:
        print(sample_config_file)
        try:
            file_object = open(sample_config_file, 'r')
            for line in file_object:
                if 'calibration_task' in line:
                    task_type = get_line_info(line)
                    # print('task_type: %s', task_type)
                    if task_type == 'camera_to_lidar':
                        camera_lidar_configs.append(sample_config_file)
                    elif task_type == 'lidar_to_gnss':
                        lidar_gnss_configs.append(sample_config_file)
                if task_type == 'camera_to_lidar':
                    for camera_lidar_task in camera_lidar_task_list:
                        if camera_lidar_task in line:
                            camera_lidar_configs.append(get_line_info(line))
                if task_type == 'lidar_to_gnss':
                    for lidar_to_gnss_task in lidar_gnss_task_list:
                        if lidar_to_gnss_task in line:
                            lidar_gnss_configs.append(get_line_info(line))
        except Exception as e:
            logging.error(e)
    return lidar_gnss_configs, camera_lidar_configs


def file_type_exist(file_dir, file_type):
    files = os.listdir(file_dir)
    # logging.info(f"files_pre:{files}")
    for k in range(len(files)):
        files[k] = os.path.splitext(files[k])[1]
    # logging.info(f"files:{files}")
    if file_type in files:
        return True
    return False


def missing_lidar_gnss_file(lidar_gnss_configs):
    if len(lidar_gnss_configs) < 3:
        return True
    logging.info(f"lidar_gnss_configs:{lidar_gnss_configs}")
    yaml_dir = os.path.dirname(lidar_gnss_configs[0])
    odometry_dir = os.path.join(yaml_dir, lidar_gnss_configs[1])
    point_cloud_dir = os.path.join(yaml_dir, lidar_gnss_configs[2])
    logging.info(f"odometry_dir:{odometry_dir} , point_cloud_dir: {point_cloud_dir}")
    if not os.access(odometry_dir, os.F_OK):
        logging.info('odometry file does not exist')
        return True
    if not file_type_exist(point_cloud_dir, '.pcd'):
        logging.info('pcd file does not exist')
        return True
    return False


def missing_camera_lidar_file(camera_lidar_configs):
    if len(camera_lidar_configs) < 4:
        return True
    yaml_dir = os.path.dirname(camera_lidar_configs[0])
    camera_lidar_pairs_dir = os.path.join(yaml_dir, camera_lidar_configs[1])
    extrinsics_yaml_dir = os.path.join(yaml_dir, camera_lidar_configs[2])
    intrinsics_yaml_dir = os.path.join(yaml_dir, camera_lidar_configs[3])
    jpg_flag = file_type_exist(camera_lidar_pairs_dir, '.jpg')
    pcd_flag = file_type_exist(camera_lidar_pairs_dir, '.pcd')
    if not (jpg_flag and pcd_flag):
        logging.info('camera_lidar_pairs data error')
        return True
    # check extrinsics.yaml
    if not os.access(extrinsics_yaml_dir, os.F_OK):
        logging.info('extrinsics_yaml file does not exist')
        return True
    # check intrinsics.yaml
    if not os.access(intrinsics_yaml_dir, os.F_OK):
        logging.info('intrinsics_yaml file does not exist')
        return True
    return False


def is_oversize_file(path):
    if file_utils.getInputDirDataSize(path) >= 1 * 1024 * 1024 * 1024:
        logging.error('The input file is oversize!')
        return True
    return False


def sanity_check(input_folder, job_owner, job_id, email_receivers=None):
    is_on_cloud = context_utils.is_cloud()
    config_flag, config_files = missing_config_file(input_folder)
    lidar_gnss_flag = False
    camera_lidar_flag = False
    lidar_gnss_configs, camera_lidar_configs = get_calibration_task_info(config_files)
    if lidar_gnss_configs:
        lidar_gnss_flag = missing_lidar_gnss_file(lidar_gnss_configs)
    if camera_lidar_configs:
        camera_lidar_flag = missing_camera_lidar_file(camera_lidar_configs)
    if is_oversize_file(input_folder):
        err_msg = "The input file is oversize(1G)!"
        if is_on_cloud:
            JobUtils(job_id).save_job_failure_code('E200')
    elif config_flag:
        err_msg = "Missing sample_config.yaml!"
        if is_on_cloud:
            JobUtils(job_id).save_job_failure_code('E201')
    elif missing_calibration_task(config_files):
        err_msg = "The sample_config.yaml file miss calibration_task config!"
        if is_on_cloud:
            JobUtils(job_id).save_job_failure_code('E202')
    elif lidar_gnss_flag and not camera_lidar_flag:
        err_msg = "Missing Lidar_gnss files!"
        if is_on_cloud:
            JobUtils(job_id).save_job_failure_code('E203')
    elif not lidar_gnss_flag and camera_lidar_flag:
        err_msg = "Missing camera_lidar files!"
        if is_on_cloud:
            JobUtils(job_id).save_job_failure_code('E204')
    elif lidar_gnss_flag and camera_lidar_flag:
        err_msg = "Missing lidar_gnss and camera_lidar files!"
        if is_on_cloud:
            JobUtils(job_id).save_job_failure_code('E205')
    else:
        logging.info("%s Passed sanity check." % input_folder)
        return True

    if is_on_cloud and email_receivers:
        title = 'Error occurred during Sensor_calibration data sanity check for {}'.format(
            job_owner)
        content = 'job_id={} input_folder={}\n{}'.format(job_id, input_folder, cgi.escape(err_msg))
        email_utils.send_email_error(title, content, email_receivers)

    logging.error(err_msg)
    return False
