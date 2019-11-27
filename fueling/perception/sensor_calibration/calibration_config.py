import os
import shutil
import yaml

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging

class CalibrationConfig(object):
    """manage the calibration config files IO
    allow user to input very simple configuration information, e.g.,
    data local path, init_extrinsics(lidar, camera), intrinsics(camera), sensor name.
    Then the class will automatically generate the complicated calibration config file
    in YAML format(as for now), to guid the calibration service.
    """
    def __init__(self, supported_calibrations=['lidar_to_gnss', 'camera_to_lidar']):
        self._task_name = 'unknown'
        self._supported_tasks = supported_calibrations
        logging.info('calibration service now support: {}'.format(self._supported_tasks))

    def _generate_lidar_to_gnss_calibration_yaml(self, root_path, result_path, in_data):
        out_data = {
            # list all input sensor messages and the file locations
            'data': {
                'odometry': os.path.abspath(os.path.join(
                        root_path, in_data['odometry_file'])),
                # 'lidars' are list of dict()
                'lidars': [
                    {
                        in_data['source_sensor']: {
                            'path': os.path.abspath(os.path.join(
                                root_path, in_data['sensor_files_directory']))+'/'
                        }
                    }
                ],
                'result': result_path,
                'calib_height': False,
                'frame': 'UTM'
            },
            # list all calibration parameters
            'calibration': {
                # extrinsics : dict of dict() for multi-lidar if needs
                # wired format. Beijing has to make the YAML consistent in multi-lidar calib.
                'init_extrinsics': {
                    in_data['source_sensor']: {
                        'translation': in_data['transform']['translation'],
                        'rotation': in_data['transform']['rotation']
                    }
                },
                #  optimization parameters: list of dict() for multi-lidar if needs
                'steps': [
                    {
                        'source_lidars': [
                            in_data['source_sensor']
                        ],
                        'target_lidars': [
                            in_data['source_sensor']
                        ],
                        'lidar_type': 'multiple',
                        'fix_target_lidars': False,
                        'fix_z': True,
                        'iteration': 3
                    }
                ]
            }
        }
        return out_data

    def _generate_camera_to_lidar_calibration_yaml(self, root_path, result_path, in_data):
        out_data = {
            # set up the source_sensor(camera name) to dest sensor(lidar name)
            'frameid': in_data['destination_sensor'],
            'child_frame_id': in_data['source_sensor'],
            # set up the inputs: intrinsic, initial extrinsic,
            # input data path for paris of lidar-camera
            'intrinsic': os.path.abspath(os.path.join(root_path, in_data['intrinsic'])),
            'extrinsic': os.path.abspath(os.path.join(root_path, in_data['extrinsic'])),
            'data_path': os.path.abspath(os.path.join(root_path, in_data['data_path'])),
            # need to specific the beam number of lidar, due to unorganized pcd input,
            # having no idea about the beam number.
            'beams': in_data['beams'],
            'out_filename': in_data['source_sensor'] + '_' + in_data['destination_sensor'] +\
                            '_extrinsics.yaml',
            'debug': True,
            'debug_path': result_path,
            # add default calibration parameters, basically not need to change
            'adjustment': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0
            },
            'scales': [0.5],
            'alpha': [0.1],
            'gamma': [0.9],
            'sld_win': [50],
            'grid_num': [5, 5, 5, 5, 5, 5],
            'grid_delta': [0.2, 0.2, 0.4, 0.05, 0.05, 0.05],
            'max_index': 6 # no idea what is this.
        }
        # copy intrinsic files to result_path folder.
        shutil.copy(out_data['intrinsic'], result_path)
        return out_data

    def get_task_name(self):
        if self._task_name == 'unknown':
            logging.error(('have not set the task name, the valid task names'
                    'are: {}'.format(self._supported_tasks)))
        return self._task_name

    def generate_task_config_yaml(self, root_path, output_path, source_config_file):
        try:
            with open(source_config_file, 'r') as f:
                data = yaml.safe_load(f)
        except:
            logging.error('cannot open the input simple configure yaml file at {}'.format(source_config_file))
            return None

        self._task_name = data['calibration_task']
        dest_config_file = os.path.join(output_path, self._task_name + '_calibration_config.yaml')
        logging.info('convert: ' + source_config_file + ' to: ' + dest_config_file)

        result_path = os.path.join(output_path, 'results')
        file_utils.makedirs(result_path)

        if not self._task_name in self._supported_tasks:
            logging.error('does not support the calibration task: {}'.format(
                        self._task_name))

        if self._task_name == 'lidar_to_gnss':
            out_data = self._generate_lidar_to_gnss_calibration_yaml(
                root_path=root_path, result_path=result_path, in_data=data)
        elif self._task_name == 'camera_to_lidar':
            out_data = self._generate_camera_to_lidar_calibration_yaml(
                root_path=root_path, result_path=result_path, in_data=data)

        logging.info(yaml.safe_dump(out_data))
        print(yaml.safe_dump(out_data))
        try:
            with  open(dest_config_file, 'w') as f:
                yaml.safe_dump(out_data, f)
        except:
            logging.error('cannot generate the task config yaml file at {}'.format(dest_config_file))
            return None
        return dest_config_file