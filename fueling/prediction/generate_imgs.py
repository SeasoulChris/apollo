#!/usr/bin/env python
import glob
import operator
import os

import colored_glog as glog
import cv2 as cv
import pyspark_utils.op as spark_op

from map_feature.obstacle_mapping import ObstacleMapping
from modules.prediction.proto import offline_features_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils


class GenerateImgs(BasePipeline):
    """generate imgs from FrameEnv."""

    def __init__(self):
        BasePipeline.__init__(self, 'generate-imgs')

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        records_dir = self.to_rdd(glob.glob('/apollo/data/prediction/features/*/frame_env.*.bin'))
        origin_prefix = '/apollo/data/prediction/frame_env'
        target_prefix = '/apollo/data/prediction/img_features'
        self.run(records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        origin_prefix = 'modules/prediction/frame_env'
        target_prefix = 'modules/prediction/img_features'
        # RDD(bin_file)
        bin_file = self.to_rdd(self.bos().list_files(origin_prefix)).filter(
            spark_op.filter_path(['*frame_env.*.bin']))
        self.run(bin_file, origin_prefix, target_prefix)

    def run(self, bin_file_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        file_list_rdd = (
            # RDD(bin_file)
            bin_file_rdd
            # PairRDD(target_dir, bin_file)
            .keyBy(lambda bin_file:
                   os.path.dirname(bin_file).replace(origin_prefix, target_prefix, 1))
            .cache())

        # Create all target_dir.
        file_list_rdd.keys().distinct().foreach(file_utils.makedirs)

        result = (
            # PairRDD(target_dir, bin_file)
            file_list_rdd
            # PairRDD(target_dir, frame_env)
            .flatMapValues(read_frame_env)
            # RDD(0/1), 1 for success
            .map(spark_op.do_tuple(self.process_frame_env))
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_frame_env(target_dir, frame_env):
        """Call prediction python code to draw images."""
        target_dir_parts = target_dir.split('/')
        region = target_dir_parts[target_dir_parts.index('img_features') + 1]
        try:
            obstacle_mapping = ObstacleMapping(region, frame_env)
            glog.debug("obstacles_history length is: " + str(len(frame_env.obstacles_history)))
            for history in frame_env.obstacles_history:
                if not history.is_trainable:
                    continue
                key = "{}@{:.3f}".format(history.feature[0].id, history.feature[0].timestamp)
                img = obstacle_mapping.crop_by_history(history)
                filename = os.path.join(target_dir, key + ".png")
                cv.imwrite(filename, img)
                glog.info('Successfuly write img to: ' + filename)
            return 1
        except BaseException:
            glog.error('Failed to process this frame.')
        return 0


def read_frame_env(file_path):
    """file_path -> FrameEnv, or None if error occurs."""
    glog.info('Read FrameEnv from {} '.format(file_path))
    try:
        list_frame = offline_features_pb2.ListFrameEnv()
        with open(file_path, 'r') as file_in:
            list_frame.ParseFromString(file_in.read())
        if len(list_frame.frame_env) > 0:
            return list_frame.frame_env
        glog.error('No message in list_frame {} or its is broken'.format(file_path))
    except Exception as e:
        glog.error('Failed to read list_frame {}: {}'.format(file_path, e))
    return []


if __name__ == '__main__':
    GenerateImgs().main()
