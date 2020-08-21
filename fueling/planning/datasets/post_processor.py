#!/usr/bin/env python
import os

from modules.planning.proto import learning_data_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.context_utils as context_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


class PostProcessor(BasePipeline):
    def run(self):
        if context_utils.is_local():
            # for Titan
            self.src_dir_prefix = '/fuel/data/output_data_evaluated'
            # self.src_dir_prefix = 'titan'
            # for data_folder in os.listdir(self.our_storage().abs_path(self.src_dir_prefix)):
            #    src_dir_prefix = os.path.join(self.src_dir_prefix, data_folder)
            #    logging.info(f'processing data folder {src_dir_prefix}')
            #    self.run_internal(src_dir_prefix)
            self.run_internal(self.src_dir_prefix)
        else:
            self.src_dir_prefix = 'modules/planning/output_data_evaluated'
            self.run_internal(self.src_dir_prefix)

    def run_internal(self, src_dir):
        bin_files = (
            # RDD(bin_files)
            self.to_rdd(self.our_storage().list_files(src_dir, '.bin'))
            # PairedRDD((dir, original_bin_file_name), bin_files)
            .keyBy(lambda src_file: (os.path.dirname(src_file), os.path.basename(src_file))))
        logging.debug(bin_files.first())

        # PairedRDD((dir, origin_file_name), data_frame))
        data_frames = bin_files.flatMapValues(self._get_data_frame)
        logging.debug(data_frames.first())

        # PairedRDD((dir, origin_file_name), ((tag, tag_id), data_frame))
        tag_data_frames = data_frames.flatMapValues(self._key_by_tag)
        logging.debug(tag_data_frames.first())

        # PairedRDD((dst_file_path, origin_file_name), data_frame)
        tag_data_frames = (
            tag_data_frames.map(
                lambda elem: self._tagged_folder(self.src_dir_prefix, elem)))
        logging.debug(tag_data_frames.keys().first())

        # write single_frame to each bin to reduce memory usage
        tagged_folders = tag_data_frames.map(self._write_single_data_frame)
        logging.debug(tagged_folders.count())

    @staticmethod
    def _write_single_data_frame(folder_filename_data, is_debug=False):
        (dst_dir, origin_filename), data = folder_filename_data
        file_utils.makedirs(dst_dir)
        logging.debug(data.message_timestamp_sec)
        file_name = f'{origin_filename}_{data.message_timestamp_sec}.bin'
        logging.info(f'{dst_dir}')
        with open(os.path.join(dst_dir, file_name), 'wb') as bin_f:
            bin_f.write(data.SerializeToString())
        if is_debug:
            txt_file_name = f'{file_name}.txt'
            proto_utils.write_pb_to_text_file(data, os.path.join(dst_dir, txt_file_name))

    @staticmethod
    def _write_data_frame(folder_data_frame, frame_len=100, is_debug=False):
        learning_data = learning_data_pb2.LearningData()
        dst_dir, data_frames = folder_data_frame
        # total_frames = len([data_frames])
        frame_count = 0
        file_count = 0
        file_utils.makedirs(dst_dir)
        logging.info(f'{dst_dir}')
        for data_frame in data_frames:
            learning_data.learning_data_frame.add().CopyFrom(data_frame)
            frame_count += 1
            if frame_count == frame_len:
                file_count += 1
                # reset frame_count
                frame_count = 0
                # write bin file
                file_name = f'{file_count}.bin'
                with open(os.path.join(dst_dir, file_name), 'wb') as bin_f:
                    bin_f.write(learning_data.SerializeToString())
                if is_debug:
                    txt_file_name = f'{file_count}.txt'
                    proto_utils.write_pb_to_text_file(learning_data.learning_data_frame[0],
                                                      os.path.join(dst_dir, txt_file_name))
                # clear learning_data
                learning_data = learning_data_pb2.LearningData()
        if len(learning_data.learning_data_frame):
            # rest data frames
            file_count += 1
            file_name = f'{file_count}.bin'
            with open(os.path.join(dst_dir, file_name), 'wb') as bin_f:
                bin_f.write(learning_data.SerializeToString())
        return file_count

    @staticmethod
    def _get_data_frame(bin_file):
        # load origin PB file
        offline_features = proto_utils.get_pb_from_bin_file(
            bin_file, learning_data_pb2.LearningData())
        # get learning data sequence
        learning_data_sequence = offline_features.learning_data_frame
        logging.debug(len(learning_data_sequence))
        return learning_data_sequence

    @staticmethod
    def _key_by_tag(data_frame):
        tag_dict = proto_utils.pb_to_dict(data_frame.planning_tag)
        tag_data_frames = []
        for key in tag_dict:
            logging.debug('{}: {}'.format(key, tag_dict[key]))
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
    def _tagged_folder(src_dir_prefix, dir_tag_data):
        (file_path, origin_file_name), ((tag, tag_id), data) = dir_tag_data
        logging.debug('file_path: {}; origin_file_name: {}'.format(file_path, origin_file_name))
        logging.debug('tag: {}; tag_id: {}'.format(tag, tag_id))

        # remove complete
        # pre_fix/record_dir/complete -> pre_fix/record_dir
        origin_file_path = os.path.split(file_path)[0]
        logging.debug('origin_file_path: {}'.format(origin_file_path))

        src_dir_elements = origin_file_path.split('/')
        dst_dir_elements = [
            'output_data_categorized' if x
            == 'output_data_evaluated' else x for x in src_dir_elements]
        dst_dir_elements.insert(-1, tag)
        dst_dir_elements.insert(-1, tag_id)
        if ('output_data_categorized' in dst_dir_elements):
            dst_dir = '/'.join(dst_dir_elements)
        else:
            dst_dir_elements = src_dir_prefix.split('/')
            while (dst_dir_elements[-1] == ''):
                dst_dir_elements.pop()
            dst_dir_elements[-1] += '_output_data_categorized'
            prefix_len = len(dst_dir_elements)
            dst_dir_elements.extend(src_dir_elements[prefix_len:])
            dst_dir_elements.append(tag)
            dst_dir_elements.append(tag_id)
            dst_dir = '/'.join(dst_dir_elements)

        return ((dst_dir, origin_file_name), data)


if __name__ == '__main__':
    PostProcessor().main()
