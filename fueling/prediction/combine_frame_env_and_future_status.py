#!/usr/bin/env python
import os
import glob
import operator
import numpy as np

from absl import flags
import pyspark_utils.op as spark_op

from modules.prediction.proto import offline_features_pb2
from modules.perception.proto import perception_obstacle_pb2

import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


TARGET_OBSTACLE_TYPE = perception_obstacle_pb2.PerceptionObstacle.PEDESTRIAN
TARGET_NUM_FUTURE_POINT = 30

'''
[scene, scene, ..., scene]
  scene: [obstacle, obstacle, ..., obstacle]
    obstacle: [history, future_status]
      history: [feature, feature, ..., feature]
        feature: [(timestamp, x, y, heading), polygon_points]
          polygon_points:[(x, y), (x, y), ..., (x, y)]
      future_status: [(x, y), (x, y), ... (x, y)]
'''
class CombineFrameEnvAndFutureStatus(BasePipeline):
    '''Records to feature proto pipeline.'''
    def run(self):
        '''Run prod.'''
        frame_env_prefix = '/fuel/kinglong_data/frame_envs/'
        if self.FLAGS.get('running_mode') == 'PROD':
            frame_env_prefix = 'modules/prediction/kinglong_frame_envs/'

        frame_env_dir = self.to_rdd(self.our_storage().list_end_dirs(frame_env_prefix))

        if frame_env_dir.isEmpty():
            logging.info('No frame env dir to be processed!')
            return

        self.run_internal(frame_env_dir)

    def run_internal(self, frame_env_dir_rdd):
        '''Run the pipeline with given arguments.'''
        result = frame_env_dir_rdd.map(self.process_dir).cache()
        # if result.isEmpty():
        #     logging.info('Nothing to be processed, everything is under control!')
        #     return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(frame_env_dir):
        logging.info(frame_env_dir)
        label_dir = frame_env_dir.replace('frame_envs', 'labels', 1)
        output_dir = frame_env_dir.replace('frame_envs', 'train', 1)
        # file_utils.makedirs(output_dir)
        mkdir_cmd = 'sudo mkdir -p {}'.format(output_dir)
        chmod_cmd = 'sudo chmod 777 {}'.format(output_dir)
        os.system(mkdir_cmd)
        logging.info(mkdir_cmd)
        os.system(chmod_cmd)
        logging.info(chmod_cmd)
        
        label_filenames = os.listdir(label_dir)
        label_dict_merged = dict()
        for label_filename in label_filenames:
            label_filepath = os.path.join(label_dir, label_filename)
            if label_filepath.endswith('future_status.npy'):
                label_dict_curr = np.load(label_filepath, allow_pickle=True).item()
                label_dict_merged.update(label_dict_curr)

        list_frame_env = offline_features_pb2.ListFrameEnv()
        frame_env_files = os.listdir(frame_env_dir)
        for frame_env_filename in frame_env_files:
            frame_envs = offline_features_pb2.ListFrameEnv()
            frame_env_filepath = os.path.join(frame_env_dir, frame_env_filename)
            frame_envs = proto_utils.get_pb_from_bin_file(frame_env_filepath, frame_envs)
            for frame_env in frame_envs.frame_env:
                list_frame_env.frame_env.append(frame_env)
        data_output = []
        for frame_env in list_frame_env.frame_env:
            scene_output = []
            for obstacle_history in frame_env.obstacles_history:
                if len(obstacle_history.feature) == 0:
                    continue
                obstacle_output = [[], []]
                for feature in obstacle_history.feature:
                    frame_output = []
                    feature_output = (feature.timestamp, feature.position.x, \
                                      feature.position.y, feature.velocity_heading)
                    frame_output.append(feature_output)
                    polygon = []
                    for point in feature.polygon_point:
                        polygon.append((point.x, point.y))
                    frame_output.append(polygon)
                    obstacle_output[0].append(frame_output)
                obstacle_output[0] = obstacle_output[0][::-1]

                obstacle_id = obstacle_history.feature[0].id
                obstacle_ts = obstacle_history.feature[0].timestamp
                obstacle_type = obstacle_history.feature[0].type
                key = '{}@{:.3f}'.format(obstacle_id, obstacle_ts)
                if obstacle_type == TARGET_OBSTACLE_TYPE and \
                   key in label_dict_merged and \
                   len(label_dict_merged[key]) > TARGET_NUM_FUTURE_POINT:
                    for i in range(1, TARGET_NUM_FUTURE_POINT + 1):
                        x = label_dict_merged[key][i][0]
                        y = label_dict_merged[key][i][1]
                        obstacle_output[1].append((x, y))
                scene_output.append(obstacle_output)
            data_output.append(scene_output)
        output_file_path = os.path.join(output_dir, 'training_data.npy')
        np.save(output_file_path, data_output)
        return 1


if __name__ == '__main__':
    CombineFrameEnvAndFutureStatus().main()
