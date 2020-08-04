#!/usr/bin/env python

from collections import Counter
import operator
import os

import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.prediction.common.configure import semantic_map_config
import fueling.common.context_utils as context_utils
import fueling.common.logging as logging
import fueling.common.spark_op as spark_op


LINEAR_ACC_THRESHOLD = 100
ANGULAR_VEL_THRESHOLD = 0.50
TURNING_ANGLE_THRESHOLD = np.pi / 6

OFFSET_X = semantic_map_config['offset_x']
OFFSET_Y = semantic_map_config['offset_y']


'''
[scene, scene, ..., scene]
  scene: [obstacle, obstacle, ..., obstacle]
    obstacle: [history, future_status]
      history: [feature, feature, ..., feature]
        feature: [timestamp, x, y, heading, polygon_points]
          polygon_points: x, y, x, y, ..., x, y # 20 (x,y)
      future_status: [x, y, x, y, ... x, y]
'''


class CleanTrainingDataPipeline(BasePipeline):
    def run(self):
        '''Run prod.'''
        training_data_prefix = ('/fuel/kinglong_data/train/' if context_utils.is_local() else
                                'modules/prediction/kinglong_train/')
        training_data_file_rdd = (
            self.to_rdd(self.our_storage()
                        .list_files(training_data_prefix))
                .filter(spark_op.filter_path(['*training_data.npy'])))
        if training_data_file_rdd.isEmpty():
            logging.info('No training data file to be processed!')
            return
        self.run_internal(training_data_file_rdd)

    def run_internal(self, training_data_file_rdd):
        '''Run the pipeline with given arguments.'''
        result = training_data_file_rdd.map(self.process_file).cache()
        # if result.isEmpty():
        #     logging.info('Nothing to be processed, everything is under control!')
        #     return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    # @staticmethod
    def process_file(self, training_data_filepath):
        logging.info(training_data_filepath)
        cleaned_training_data_filepath = training_data_filepath.replace('train', 'train_clean', 1)
        cleaned_training_data_dir = os.path.dirname(cleaned_training_data_filepath)
        mkdir_cmd = 'sudo mkdir -p {}'.format(cleaned_training_data_dir)
        chmod_cmd = 'sudo chmod 777 {}'.format(cleaned_training_data_dir)
        os.system(mkdir_cmd)
        logging.info(mkdir_cmd)
        os.system(chmod_cmd)
        logging.info(chmod_cmd)
        self.CleanTrainingData(training_data_filepath, cleaned_training_data_filepath)
        return 1

    '''
    @param future_sequence: list [x, y, x, y, ... x, y]
    '''

    def CleanFutureSequence(self, future_sequence, pred_len=30):
        # 1. Only keep pred_len length
        if len(future_sequence) < 2 * pred_len:
            return None
        future_len = len(future_sequence)
        obs_pos = (np.array(future_sequence)).reshape([future_len // 2, 2])
        obs_pos = obs_pos - obs_pos[0, :]
        # 2. Get the scalar acceleration, and angular speed of all points.
        obs_vel = (obs_pos[1:, :] - obs_pos[:-1, :]) / 0.1
        linear_vel = np.linalg.norm(obs_vel, axis=1)
        linear_acc = (linear_vel[1:] - linear_vel[0:-1]) / 0.1
        angular_vel = np.sum(
            obs_vel[1:, :] * obs_vel[:-1, :], axis=1) / ((linear_vel[1:] * linear_vel[:-1]) + 1e-6)
        turning_ang = (
            np.arctan2(obs_vel[-1, 1], obs_vel[-1, 0])
            - np.arctan2(obs_vel[0, 1], obs_vel[0, 0])) % (2 * np.pi)
        turning_ang = turning_ang if turning_ang < np.pi else turning_ang - 2 * np.pi
        # 3. Filtered the extream values for acc and ang_vel.
        if np.max(np.abs(linear_acc)) > LINEAR_ACC_THRESHOLD:
            return None
        if np.min(angular_vel) < ANGULAR_VEL_THRESHOLD:
            return None
        # Get the statistics of the cleaned labels, and do some re-balancing to
        # maintain roughly the same distribution as before.
        if turning_ang < -TURNING_ANGLE_THRESHOLD:
            return 'right'
        elif TURNING_ANGLE_THRESHOLD < turning_ang:
            return 'left'
        return 'straight'

    '''
    @param training_data_dir: end dir containing training_data.npy
    '''

    def CleanTrainingData(self, training_data_filepath, cleaned_training_data_filepath):
        count = Counter()
        file_content = np.load(training_data_filepath, allow_pickle=True).tolist()
        cleaned_content = []
        region = self.get_map_dir_by_path(training_data_filepath)
        for scene in file_content:
            if self.SceneHasInvalidDataPt(scene, region):
                continue
            cleaned_scene = []
            num_valid_label = 0
            for data_pt in scene:
                cleaned_data_pt = []
                cleaned_data_pt.append(data_pt[0])
                turn_type = self.CleanFutureSequence(data_pt[1])
                if turn_type is None:
                    cleaned_data_pt.append([])
                else:
                    cleaned_data_pt.append(data_pt[1])
                    count[turn_type] += 1
                    num_valid_label += 1
                cleaned_data_pt.append(data_pt[2])
                cleaned_scene.append(cleaned_data_pt)
            if num_valid_label > 0:
                cleaned_content.append(cleaned_scene)
        logging.info(count)
        logging.info('npy save {}'.format(cleaned_training_data_filepath))
        np.save(cleaned_training_data_filepath, cleaned_content)

    def SceneHasInvalidDataPt(self, scene, region):
        for data_pt in scene:
            if len(data_pt[1]) == 0:
                continue
            if len(data_pt[0]) == 0:
                return True
            if self.IsDataPtZeroCurrPosition(data_pt):
                return True
            if self.IsDataPtOutsideMapArea(data_pt, region):
                return True
        return False

    def IsDataPtZeroCurrPosition(self, data_pt):
        # curr = data_pt[0][-1]
        curr_x = data_pt[0][-1][1] + OFFSET_X
        curr_y = data_pt[0][-1][2] + OFFSET_Y
        if abs(curr_x) < 1.0 or abs(curr_y) < 1.0:
            return True
        return False

    def IsDataPtOutsideMapArea(self, data_pt, map_region, threshold=10.0, history_limit=10):
        region_param = semantic_map_config['map_coords'][map_region]
        buff = region_param['distance_buffer']
        x_range = region_param['horizontal_pixel_size'] * region_param['resolution']
        y_range = region_param['vertical_pixel_size'] * region_param['resolution']
        accepted_x_lower = region_param['lower_left_x'] + buff - threshold
        accepted_y_lower = region_param['lower_left_y'] + buff - threshold
        accepted_x_upper = region_param['lower_left_x'] + x_range - buff + threshold
        accepted_y_upper = region_param['lower_left_y'] + y_range - buff + threshold
        history = data_pt[0]
        length = min(len(history), history_limit)
        start_index = len(history) - length
        for i in range(start_index, len(history)):
            x = history[i][1] + OFFSET_X
            y = history[i][2] + OFFSET_Y
            if x < accepted_x_lower or x > accepted_x_upper or \
               y < accepted_y_lower or y > accepted_y_upper:
                return True
        return False

    def get_map_dir_by_path(self, path):
        path_lower = path.lower()
        if "baidudasha" in path_lower:
            return "baidudasha"
        if "xionganshiminzhongxin" in path_lower:
            return "XiongAn"
        if "xiamen" in path_lower:
            return "XiaMen"
        if "feifengshan" in path_lower:
            return "FuZhouFeiFengShan"
        return "demo"


if __name__ == '__main__':
    CleanTrainingDataPipeline().main()
