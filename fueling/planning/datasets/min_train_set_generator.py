#!/usr/bin/env python
import os
from shutil import copyfile

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils


class MinTrainSetGenerator(BasePipeline):
    def run(self):
        # ratio should not be 0
        self.down_sampling_ratio = 1
        self.train_validation_test_ratio = {'train': 0.81,
                                            'to_be_synthesized': 0.09,
                                            'validation': 0.05,
                                            'test': 0.05}
        if self.is_local():
            # for Titan
            self.src_dir = 'data/output_data_balanced/lane_turn'
            self.dst_dir = 'data/min_train_set/lane_turn'
            self.log_file = '/fuel/data/min_train_set/lane_turn/log.txt'
            file_utils.makedirs(os.path.dirname(self.log_file))
            logging.info(f'processing data folder {self.src_dir}')
            self.run_internal(self.src_dir)

    def run_internal(self, src_dir):
        logging.info(f'down_sampling_ratio: {self.down_sampling_ratio}')
        logging.info(f'training set ratio: {self.train_validation_test_ratio}')
        with open(self.log_file, 'a+') as f:
            f.write(f'down_sampling_ratio: {self.down_sampling_ratio}\n')
            f.write(f'training set ratio: {self.train_validation_test_ratio}\n')
        bin_files = self.to_rdd(self.our_storage().list_files(src_dir, '.bin'))
        if self.down_sampling_ratio < 1:
            bin_files = bin_files.sample(False, self.down_sampling_ratio)

        ratio = [self.train_validation_test_ratio[key]
                 for key in self.train_validation_test_ratio]
        logging.info(ratio)
        bin_files_dict = {}
        (bin_files_dict['train'],
         bin_files_dict['to_be_synthesized'],
         bin_files_dict['validation'],
         bin_files_dict['test']) = bin_files.randomSplit(ratio, 4)

        for key in bin_files_dict:
            bin_files = bin_files_dict[key]
            dst_dir = os.path.join(self.dst_dir, key)
            # copy the bin files to target path
            target_bin_files = bin_files.map(
                lambda elem: self._copy_bin_files(
                    elem,
                    src_dir,
                    dst_dir))
            file_count = target_bin_files.count()
            logging.info(f'{key} sample count: {file_count}')
            with open(self.log_file, 'a+') as f:
                f.write(f'In path: {dst_dir} \n')
                f.write(f'{key} sample count: {file_count}\n')

    @staticmethod
    def _copy_bin_files(src_file_path, src_dir_prefix, dst_dir_prefix):
        dst_file_path = src_file_path.replace(src_dir_prefix, dst_dir_prefix)
        logging.debug(dst_file_path)
        dst_dir = os.path.dirname(dst_file_path)
        file_utils.makedirs(dst_dir)
        copyfile(src_file_path, dst_file_path)
        return dst_file_path


if __name__ == '__main__':
    MinTrainSetGenerator().main()
