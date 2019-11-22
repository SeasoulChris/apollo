import os
import yaml

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging

supported_calibrations = ['lidar_to_gnss']
class CalibrationConfig(object):
	"""manage the calibration config files IO
	allow user to input very simple configuration information, e.g.,
	data local path, init_extrinsics(lidar, camera), intrinsics(camera), sensor name.
	Then the class will automatically generate the complicated calibration config file
	in YAML format(as for now), to guid the calibration service.
	"""
	def __init__(self, task_name):
		self._task_name = task_name
		if not self._task_name in supported_calibrations:
			logging.error('does not support the calibration task: {}'.format(
							self._task_name))

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
                        'translation': in_data['translation'],
                        'rotation': in_data['rotation']
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

	def generate_task_config_yaml(self, source_config_file, dest_config_file, root_path):
		logging.info('convert: '+source_config_file+ ' to: '+dest_config_file)
		try:
			with open(source_config_file, 'r') as f:
				data = yaml.safe_load(f)
		except:
			logging.error('cannot open the input simple configure yaml file at {}'.format(source_config_file))
			return

		result_path = os.path.join(root_path, 'results')
		file_utils.makedirs(result_path)

		if self._task_name == 'lidar_to_gnss':
			out_data = self._generate_lidar_to_gnss_calibration_yaml(
				root_path=root_path, result_path=result_path, in_data=data)

			logging.info(yaml.safe_dump(out_data))
			print(yaml.safe_dump(out_data))
			try:
				with  open(dest_config_file, 'w') as f:
					yaml.safe_dump(out_data, f)
			except:
				logging.error('cannot generate the task config yaml file at {}'.format(dest_config_file))
		return
        # out_data = {
	        #     # list all input sensor messages and the file locations
	        #     'data': {
	        #         'odometry': os.path.abspath(os.path.join(
	        #                 original_path, data['odometry_file'])),
	        #         # 'lidars' are list of dict()
	        #         'lidars': [
	        #             {
	        #                 data['source_sensor']: {
	        #                     'path': os.path.abspath(os.path.join(
	        #                         original_path, data['sensor_files_directory']))+'/'
	        #                 }
	        #             }
	        #         ],
	        #         'result': result_path,
	        #         'calib_height': False,
	        #         'frame': 'UTM' 
	        #     },
	        #     # list all calibration parameters
	        #     'calibration': {
	        #         # extrinsics : dict of dict() for multi-lidar if needs
	        #         # wired format. Beijing has to make the YAML consistent in multi-lidar calib.
	        #         'init_extrinsics': {
	        #             data['source_sensor']: {
	        #                 'translation': data['translation'],
	        #                 'rotation': data['rotation']
	        #             }
	        #         },
	        #         #  optimization parameters: list of dict() for multi-lidar if needs
	        #         'steps': [
	        #             {
	        #                 'source_lidars': [
	        #                     data['source_sensor']
	        #                 ],
	        #                 'target_lidars': [
	        #                     data['source_sensor']
	        #                 ],
	        #                 'lidar_type': 'multiple',
	        #                 'fix_target_lidars': False,
	        #                 'fix_z': True,
	        #                 'iteration': 3
	        #             }
	        #         ]
	        #     }
	        # }