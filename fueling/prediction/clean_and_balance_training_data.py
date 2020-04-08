#!/usr/bin/env python

from collections import Counter
import glob
import operator
import os

from absl import flags
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging

LINEAR_ACC_THRESHOLD = 50
ANGULAR_VEL_THRESHOLD = 0.85
TURNING_ANGLE_THRESHOLD = np.pi/6

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
        training_data_prefix = '/fuel/kinglong_data/train/'
        if self.FLAGS.get('running_mode') == 'PROD':
            training_data_prefix = 'modules/prediction/kinglong_train/'
        training_data_dir_rdd = self.to_rdd(self.our_storage().list_end_dirs(training_data_prefix))
        if training_data_dir_rdd.isEmpty():
            logging.info('No training data dir to be processed!')
            return
        self.run_internal(training_data_dir_rdd)

    def run_internal(self, training_data_dir_rdd):
        '''Run the pipeline with given arguments.'''
        result = training_data_dir_rdd.map(self.process_dir).cache()
        # if result.isEmpty():
        #     logging.info('Nothing to be processed, everything is under control!')
        #     return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    # @staticmethod
    def process_dir(self, training_data_dir):
        logging.info(training_data_dir)
        cleaned_training_data_dir = training_data_dir.replace('train', 'train_clean', 1)
        mkdir_cmd = 'sudo mkdir -p {}'.format(cleaned_training_data_dir)
        chmod_cmd = 'sudo chmod 777 {}'.format(cleaned_training_data_dir)
        os.system(mkdir_cmd)
        logging.info(mkdir_cmd)
        os.system(chmod_cmd)
        logging.info(chmod_cmd)
        self.CleanTrainingData(training_data_dir, cleaned_training_data_dir)
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
        turning_ang = (np.arctan2(
            obs_vel[-1, 1], obs_vel[-1, 0]) - np.arctan2(obs_vel[0, 1], obs_vel[0, 0])) % (2*np.pi)
        turning_ang = turning_ang if turning_ang < np.pi else turning_ang-2*np.pi
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
    def CleanTrainingData(self, training_data_dir, cleaned_training_data_dir):
        training_data_filenames = os.listdir(training_data_dir)
        count = Counter()
        for filename in training_data_filenames:
            filepath = os.path.join(training_data_dir, filename)
            if 'training_data.npy' not in filepath:
                    continue
            file_content = np.load(filepath, allow_pickle=True).tolist()
            cleaned_content = []
            for scene in file_content:
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
                    cleaned_scene.append(cleaned_data_pt)
                if num_valid_label > 0:
                    cleaned_content.append(cleaned_scene)
            logging.info(count)
            cleaned_filepath = os.path.join(cleaned_training_data_dir, 'training_data_clean.npy')
            logging.info('npy save {}'.format(cleaned_filepath))
            np.save(cleaned_filepath, cleaned_content)


if __name__ == '__main__':
    CleanTrainingDataPipeline().main()
