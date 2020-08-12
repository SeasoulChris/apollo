#!/usr/bin/env python
from absl import flags

from fueling.common.base_pipeline import SequentialPipeline
from fueling.prediction.copy_map import CopyMap
from fueling.prediction.dump_feature_proto import DumpFeatureProto
from fueling.prediction.generate_labels import GenerateLabels
from fueling.prediction.frame_env import FrameEnv
from fueling.prediction.combine_frame_env_and_future_status import CombineFrameEnvAndFutureStatus
from fueling.prediction.learning.pipelines.pedestrian_trajectory_prediction \
    .pedestrian_trajectory_training import PedestrianTraining


flags.DEFINE_string('input_path', '/fuel/kinglong_data/', 'Iput data path')


if __name__ == '__main__':
    SequentialPipeline([
        CopyMap(),
        DumpFeatureProto(),
        GenerateLabels(),
        FrameEnv(),
        CombineFrameEnvAndFutureStatus(),
        PedestrianTraining()
    ]).main()
