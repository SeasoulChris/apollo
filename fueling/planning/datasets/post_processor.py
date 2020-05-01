#!/usr/bin/env python
import operator
import os
import time

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging

from modules.planning.proto import learning_data_pb2
import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils


class PostProcessor(BasePipeline):
    def run(self):
        if self.is_local():
            # for Titan
            self.src_dir_prefix = 'data/output_data_evaluated'
            self.dst_dir_prefix = 'data/output_data_categorized'
            # self.src_dir_prefix = 'titan'
            # self.dst_dir_prefix = 'apollo/data/output_data'
            for data_folder in os.listdir(self.our_storage().abs_path(self.src_dir_prefix)):
                src_dir_prefix = os.path.join(self.src_dir_prefix, data_folder)
                logging.info(f'processing data folder {src_dir_prefix}')
                self.run_internal(src_dir_prefix)
        else:
            self.src_dir_prefix = 'modules/planning/output_data_evaluated'
            self.dst_dir_prefix = 'modules/planning/output_data_categorized'
            self.run_internal(self.src_dir_prefix)

    def run_internal(self, src_dir):
        logging.info(src_dir)
        bin_files = (
            # RDD(bin_files)
            self.to_rdd(self.our_storage().list_files(src_dir, '.bin'))
            # PairedRDD(dir, bin_files)
            .keyBy(lambda src_file: os.path.dirname(src_file)))
        todo_bin_files = bin_files
        logging.info(bin_files.first())
        logging.info(todo_bin_files.count())
        # PairedRDD(dir,data_frame)
        data_frames = bin_files.flatMapValues(self._get_data_frame)
        logging.info(data_frames.count())
        logging.debug(data_frames.first())
        # PairedRDD(dir, ((tag, tag_id), data_frame))
        tag_data_frames = data_frames.flatMapValues(self._key_by_tag)
        logging.info(tag_data_frames.count())
        logging.debug(tag_data_frames.first())
        tag_data_frames = (tag_data_frames
                           # remove 'complete' from path_dir
                           # pre_fix/record_dir/complete -> pre_fix/record_dir
                           # PairedRDD(dst_file_path, ((tag, tag_id), data_frame))
                           .map(lambda elem: (os.path.split(elem[0])[0], elem[1]))
                           # pre_fix/record_dir/complete -> dst_fix/record_dir
                           # PairedRDD(dst_file_path, ((tag, tag_id), data_frame))
                           .map(lambda elem: (elem[0].replace(self.src_dir_prefix, self.dst_dir_prefix), elem[1]))
                           #PairedRDD(dst_file_path / tag / tag_id, data_frame)
                           .map(self._tagged_folder))
        logging.info(tag_data_frames.count())
        logging.debug(tag_data_frames.first())
        logging.info(tag_data_frames.keys().first())
        # collect data according to folder
        if self.is_local():
            # write single_frame to each bin to reduce memory usage
            tagged_folders = tag_data_frames.map(self._write_single_data_frame)
        else:
            tagged_folders = tag_data_frames.groupByKey().map(self._write_data_frame)
        logging.info(tagged_folders.count())

    @staticmethod
    def _write_single_data_frame(folder_data_frame):
        learning_data = learning_data_pb2.LearningData()
        dst_dir, data_frames = folder_data_frame
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        logging.debug(data_frames.message_timestamp_sec)
        file_name = f'{data_frames.message_timestamp_sec}.bin'
        with open(os.path.join(dst_dir, file_name), 'wb') as bin_f:
            bin_f.write(learning_data.SerializeToString())

    @staticmethod
    def _write_data_frame(folder_data_frame, frame_len=100, is_debug=False):
        learning_data = learning_data_pb2.LearningData()
        dst_dir, data_frames = folder_data_frame
        total_frames = len([data_frames])
        frame_count = 0
        file_count = 0
        file_utils.makedirs(dst_dir)
        for data_frame in data_frames:
            learning_data.learning_data.add().CopyFrom(data_frame)
            frame_count += 1
            if frame_count == frame_len:
                file_count += 1
                # write bin file
                file_name = f'{file_count}.bin'
                with open(os.path.join(dst_dir, file_name), 'wb') as bin_f:
                    bin_f.write(learning_data.SerializeToString())
                if is_debug:
                    txt_file_name = f'{file_count}.txt'
                    proto_utils.write_pb_to_text_file(learning_data.learning_data[0],
                                                      os.path.join(dst_dir, txt_file_name))
                # clear learning_data
                learning_data = learning_data_pb2.LearningData()
        if len(learning_data.learning_data):
            # rest data frames
            file_count += 1
            file_name = f'{file_count}.bin'
            with open(os.path.join(dst_dir, file_name), 'wb') as bin_f:
                bin_f.write(learning_data.SerializeToString())
            # for debug only
            # txt_file_name = f'{file_count}.txt'
            # proto_utils.write_pb_to_text_file(learning_data.learning_data[0],
            #                                   os.path.join(dst_dir, txt_file_name))
        return file_count

    @staticmethod
    def _get_data_frame(bin_file):
        # load origin PB file
        offline_features = proto_utils.get_pb_from_bin_file(
            bin_file, learning_data_pb2.LearningData())
        # get learning data sequence
        learning_data_sequence = offline_features.learning_data
        logging.debug(len(learning_data_sequence))
        return learning_data_sequence

    @staticmethod
    def _key_by_tag(data_frame):
        tag_dict = proto_utils.pb_to_dict(data_frame.planning_tag)
        tag_data_frames = []
        for key in tag_dict:
            logging.debug(key)
            logging.debug(tag_dict[key])
            logging.debug(len(tag_dict[key]))
            if len(tag_dict[key]) == 2:
                # overlap features
                tag_id = tag_dict[key]['id']
                logging.debug(tag_id)
            else:
                # laneturn feature
                tag_id = tag_dict[key]
            tag_data_frames.append(((key, tag_id), data_frame))
        return tag_data_frames

    @staticmethod
    def _tagged_folder(tag_data_frames):
        file_path, ((tag, tag_id), data_frame) = tag_data_frames
        logging.debug(tag)
        # id & distance
        logging.debug(tag_id)
        dst_file_path = os.path.join(file_path, tag, tag_id)
        return (dst_file_path, data_frame)


if __name__ == '__main__':
    PostProcessor().main()
