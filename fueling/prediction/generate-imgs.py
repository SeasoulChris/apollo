#!/usr/bin/env python
import fnmatch
import glob
import operator
import os

import cv2 as cv
import pyspark_utils.op as spark_op

from map_feature.obstacle_mapping import ObstacleMapping
from modules.prediction.proto import offline_features_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class GenerateImgs(BasePipeline):
    """generate imgs from FrameEnv."""
    def __init__(self):
        BasePipeline.__init__(self, 'generate-imgs')

    def run_test(self):
        """Run test."""
        sc = self.get_spark_context()
        root_dir = '/apollo'
        # RDD(dir_path)
        records_dir = sc.parallelize(glob.glob('/apollo/data/prediction/features/*/frame_env.*.bin'))
        origin_prefix = 'data/prediction/features'
        target_prefix = 'data/prediction/img_features'
        self.run(root_dir, records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        bucket = 'apollo-platform'
        origin_prefix = 'data/prediction/features'
        target_prefix = 'data/prediction/img_features'

        bin_file = (
            # RDD(file), start with origin_prefix
            s3_utils.list_files(bucket, origin_prefix)
            # RDD(bin_files)
            .filter(lambda src_file: fnmatch.fnmatch(src_file, '*frame_env.*.bin'))
            # RDD(bin_files), which is unique
            .distinct())
        self.run(root_dir, bin_file, origin_prefix, target_prefix)

    def run(self, root_dir, bin_file_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        file_list_rdd = (
            # RDD(bin_file)
            bin_file_rdd
            # PairRDD(target_dir, bin_file)
            .map(lambda bin_file: (os.path.dirname(bin_file).replace(origin_prefix, target_prefix, 1),
                                   bin_file))
            # PairRDD(target_dir, bin_file), in absolute path
            .map(lambda src_dst: (os.path.join(root_dir, src_dst[0]),
                                  os.path.join(root_dir, src_dst[1]))))
        (
            # PairRDD(target_dir, bin_file), in absolute path
            file_list_rdd
            # RDD(target_dir), in absolute path
            .keys()
            # RDD(target_dir)
            .distinct()
            # makedirs for all target_dir
            .foreach(file_utils.makedirs))
        result = (
            # PairRDD(target_dir, bin_file), in absolute path
            file_list_rdd
            # PairRDD(target_dir, frame_env_list), in absolute path
            .mapValues(lambda bin_file: read_frame_env(bin_file))
            # PairRDD(target_dir, frame_env), in absolute path
            .flatMapValues(lambda frame_env: frame_env)
            # RDD(0/1), 1 for success
            .map(spark_op.do_tuple(self.process_frame_env))
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_frame_env(target_dir, frame_env):
        """Call prediction python code to draw images."""
        region = target_dir.split('/')[target_dir.split('/').index('img_features') + 1]
        try:
            obstacle_mapping = ObstacleMapping(region, frame_env)
            glog.info(len(frame_env.obstacles_history))
            for history in frame_env.obstacles_history:
                if not history.is_trainable:
                    continue
                key = "{}@{:.3f}".format(history.feature[0].id, history.feature[0].timestamp)
                img = obstacle_mapping.crop_by_history(history)
                filename = os.path.join(target_dir, key + ".png")
                cv.imwrite(filename, img)
                glog.info('Successfuly write img to: ' + filename)
            return 1
        except:
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
    GenerateImgs().run_prod()
