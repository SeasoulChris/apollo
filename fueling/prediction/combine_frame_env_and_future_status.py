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
from fueling.prediction.common.configure import semantic_map_config


TARGET_OBSTACLE_TYPE = perception_obstacle_pb2.PerceptionObstacle.PEDESTRIAN
TARGET_NUM_FUTURE_POINT = 30
MAX_NUM_LABEL_FILE = 180
MAX_NUM_FRAME_ENV_FILE = 100

OFFSET_X = semantic_map_config['offset_x']
OFFSET_Y = semantic_map_config['offset_y']

'''
[scene, scene, ..., scene]
  scene: [obstacle, obstacle, ..., obstacle]
    obstacle: [history, future_status, id]
      history: [feature, feature, ..., feature]
        feature: [timestamp, x, y, heading, polygon_points]
          polygon_points: x, y, x, y, ..., x, y # 20 (x,y)
      future_status: [x, y, x, y, ... x, y]
      id: [obstacle_id]
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

    def GetObstacleOutput(self, obstacle_history, label_dict_merged):
        if len(obstacle_history.feature) == 0:
            return None
        obstacle_output = [[], [], []]
        for feature in obstacle_history.feature:
            frame_output = [0 for i in range(4 + 20 * 2)]
            frame_output[0] = feature.timestamp
            frame_output[1] = feature.position.x - OFFSET_X
            frame_output[2] = feature.position.y - OFFSET_Y
            frame_output[3] = feature.velocity_heading
            i = 4
            for point in feature.polygon_point:
                if i >= len(frame_output):
                    break
                frame_output[i] = point.x - OFFSET_X
                i += 1
                frame_output[i] = point.y - OFFSET_Y
                i += 1
            obstacle_output[0].append(frame_output)
        obstacle_output[0] = obstacle_output[0][::-1]

        obstacle_id = obstacle_history.feature[0].id
        obstacle_ts = obstacle_history.feature[0].timestamp
        obstacle_type = obstacle_history.feature[0].type
        obstacle_output[2].append(obstacle_id)
        key = '{}@{:.3f}'.format(obstacle_id, obstacle_ts)
        if obstacle_type == TARGET_OBSTACLE_TYPE and \
           key in label_dict_merged and \
           len(label_dict_merged[key]) > TARGET_NUM_FUTURE_POINT:
            for i in range(1, TARGET_NUM_FUTURE_POINT + 1):
                x = label_dict_merged[key][i][0]
                y = label_dict_merged[key][i][1]
                obstacle_output[1].append(x - OFFSET_X)
                obstacle_output[1].append(y - OFFSET_Y)
        return obstacle_output

    def process_dir(self, frame_env_dir):
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
        label_file_start = 0
        label_file_end = len(label_filenames)
        if len(label_filenames) > MAX_NUM_LABEL_FILE:
            label_file_start = (len(label_filenames) - MAX_NUM_LABEL_FILE) // 2
            label_file_end = label_file_start + MAX_NUM_LABEL_FILE
        logging.info('label [start, end) = [{}, {})'.format(label_file_start, label_file_end))
        label_dict_merged = dict()
        for label_filename in label_filenames:
            file_index = int(label_filename.split('.')[1])
            logging.info('label file index ({})'.format(file_index))
            if file_index < label_file_start or file_index >= label_file_end:
                continue
            label_filepath = os.path.join(label_dir, label_filename)
            if label_filepath.endswith('future_status.npy'):
                label_dict_curr = np.load(label_filepath, allow_pickle=True).item()
                label_dict_merged.update(label_dict_curr)
        logging.info('Finished merge labels from {}, total {}'.format(label_dir,
                                                                      len(label_dict_merged)))

        data_output = []
        frame_env_files = os.listdir(frame_env_dir)
        start = 0
        end = len(frame_env_files)
        if len(frame_env_files) > MAX_NUM_FRAME_ENV_FILE:
            start = (len(frame_env_files) - MAX_NUM_FRAME_ENV_FILE) // 2
            end = start + MAX_NUM_FRAME_ENV_FILE
        logging.info('frame_env [start, end) = [{}, {})'.format(start, end))
        for frame_env_filename in frame_env_files:
            file_index = int(frame_env_filename.split('.')[-2])
            logging.info('file_index = {}'.format(file_index))
            if file_index < start or file_index >= end:
                continue
            frame_envs = offline_features_pb2.ListFrameEnv()
            frame_env_filepath = os.path.join(frame_env_dir, frame_env_filename)
            frame_envs = proto_utils.get_pb_from_bin_file(frame_env_filepath, frame_envs)
            logging.info('dealing with frame env file {}'.format(frame_env_filepath))
            for frame_env in frame_envs.frame_env:
                scene_output = []
                if frame_env.ego_history is None:
                    continue
                ego_output = self.GetObstacleOutput(frame_env.ego_history, label_dict_merged)
                if ego_output is None:
                    continue
                scene_output.append(ego_output)

                for obstacle_history in frame_env.obstacles_history:
                    obstacle_output = self.GetObstacleOutput(obstacle_history, label_dict_merged)
                    if obstacle_output is None:
                        continue
                    scene_output.append(obstacle_output)

                has_label = False
                for obs in scene_output:
                    if len(obs[1]) > 0:
                        has_label = True
                        break
                if not has_label:
                    continue

                # TODO(kechxu) this is just a patch, fix it from root
                has_invalid_position = False
                for obs in scene_output:
                    if abs(obs[0][-1][1] + OFFSET_X) < 1.0 or \
                       abs(obs[0][-1][2] + OFFSET_Y) < 1.0:
                        has_invalid_position = True
                        break
                if has_invalid_position:
                    continue

                data_output.append(scene_output)

            logging.info('So far, data_output length = {}'.format(len(data_output)))

        output_file_path = os.path.join(output_dir, 'training_data.npy')
        if len(data_output) > 0:
            logging.info('length of data_output is {}'.format(len(data_output)))
            np.save(output_file_path, data_output)
            logging.info('npy saved {}'.format(output_file_path))
        else:
            logging.info('Skip saving empty data {}'.format(output_file_path))
        return 1


if __name__ == '__main__':
    CombineFrameEnvAndFutureStatus().main()
