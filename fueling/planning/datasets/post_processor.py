#!/usr/bin/env python
import operator
import os
import time

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging

from modules.planning.proto import learning_data_pb2
import fueling.common.proto_utils as proto_utils


class PostProcessor(BasePipeline):
    def run(self):
        if self.is_local():
            self.src_dir_prefix = 'apollo/data/post_processor'
            self.dst_dir_prefix = 'apollo/data/output_data'
        else:
            self.src_dir_prefix = 'modules/planning/learning_data'
            self.dst_dir_prefix = 'modules/planning/output_data'

        bin_files = (
            # RDD(files)
            self.to_rdd(self.our_storage().list_files(self.src_dir_prefix))
            # RDD(bin_files)
            .filter(spark_op.filter_path(['*.bin']))
            # PairedRDD(dir, bin_files)
            .keyBy(lambda src_file: os.path.dirname(src_file)))
        todo_bin_files = bin_files
        logging.info(bin_files.first())
        logging.info(todo_bin_files.count())
        # PairedRDD(dir,data_frame)
        data_frames = bin_files.flatMapValues(self._get_data_frame)
        logging.info(data_frames.count())
        # logging.info(data_frames.first())
        # PairedRDD(dir, ((tag, tag_id), data_frame))
        tag_data_frames = data_frames.flatMapValues(self._key_by_tag)
        logging.info(tag_data_frames.count())
        # logging.info(tag_data_frames.first())
        # write data frame to folder
        tag_data_frames = (tag_data_frames
                           # PairedRDD(dst_file_path, ((tag, tag_id), data_frame))
                           .map(lambda elem: (elem[0].replace(
                               self.src_dir_prefix, self.dst_dir_prefix), elem[1]))
                           .map(self._tagged_folder))
        logging.info(tag_data_frames.count())
        logging.info(tag_data_frames.first())

    @staticmethod
    def _get_data_frame(bin_file):
        # load origin PB file
        offline_features = proto_utils.get_pb_from_bin_file(
            bin_file, learning_data_pb2.LearningData())
        # get learning data sequence
        learning_data_sequence = offline_features.learning_data
        logging.info(len(learning_data_sequence))
        return learning_data_sequence

    @staticmethod
    def _key_by_tag(data_frame):
        # logging.info(data_frame.planning_tag)
        tag_bp = learning_data_pb2.PlanningTag()
        tag_dict = proto_utils.pb_to_dict(data_frame.planning_tag)
        tag_data_frames = []
        for key in tag_dict:
            logging.info(key)
            logging.info(tag_dict[key])
            tag_data_frames.append(((key, tag_dict[key]), data_frame))
        return tag_data_frames

    @staticmethod
    def _tagged_folder(tag_data_frames):
        file_path, ((tag, tag_id), data_frame) = tag_data_frames
        logging.info(tag)
        # id & distance
        logging.info(tag_id)
        planning_tag = tag + tag_id
        dst_file_path = os.path.join(file_path, planning_tag)
        return (dst_file_path, data_frame)


if __name__ == '__main__':
    PostProcessor().main()
