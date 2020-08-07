#!/usr/bin/env python
import os
import shutil

from fueling.common.base_pipeline import SequentialPipeline
import fueling.common.context_utils as context_utils
from fueling.prediction.dump_feature_proto import DumpFeatureProto
from fueling.prediction.generate_labels import GenerateLabels
from fueling.prediction.frame_env import FrameEnv
from fueling.prediction.combine_frame_env_and_future_status import CombineFrameEnvAndFutureStatus

if __name__ == '__main__':
    input_path = '/fuel/kinglong_data/'  # TODO(all) make it configurable
    
    records_path = os.path.join(input_path, 'records')
    map_path = os.path.join(input_path, 'map/')
    map_target_path = 'code/apollo_map/'  # TODO(all) verify if it works
    if context_utils.is_local():  
        map_target_path = '/apollo/modules/map/data/'
    copy_map_command = 'cp -r {}* {}'.format(map_path, map_target_path)
    os.system(copy_map_command)  # TODO(all) verify if it works

    labels_path = records_path.replace('records', 'labels')
    frame_envs_path = records_path.replace('records', 'frame_envs')
    SequentialPipeline([
        DumpFeatureProto(records_path, labels_path),
        GenerateLabels(labels_path),
        FrameEnv(records_path, frame_envs_path),
        CombineFrameEnvAndFutureStatus(frame_envs_path),
    ]).main()
