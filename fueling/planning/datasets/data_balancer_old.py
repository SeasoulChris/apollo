import os
import time
from shutil import copyfile

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils


class DataBalancer(BasePipeline):
    def run(self):
        # ratio should not be 0
        self.ratio = {'LEFT_TURN': 0.01,
                      'RIGHT_TURN': 0.04,
                      'U_TURN': 0.005,
                      'NO_TURN': 0.945}
        self.frame_nums = {'LEFT_TURN': 0,
                           'RIGHT_TURN': 0,
                           'U_TURN': 0,
                           'NO_TURN': 0}
        if self.is_local():
            # for Titan
            self.src_dir_prefix = 'data/output_data_categorized/lane_turn'
            self.dst_dir_prefix = 'data/output_data_balanced/lane_turn'
            self.log_file = '/fuel/data/output_data_balanced/data_balancer.log'
            file_utils.makedirs(os.path.dirname(self.log_file))
            logging.info(f'processing data folder {self.src_dir_prefix}')
            self.run_internal(self.src_dir_prefix)

    def run_internal(self, src_dir_prefix):
        logging.info(src_dir_prefix)
        bin_files_dict = {}

        for data_folder in os.listdir(self.our_storage().abs_path(src_dir_prefix)):
            src_dir = os.path.join(src_dir_prefix, data_folder)
            bin_files = self.to_rdd(self.our_storage().list_files(src_dir, '.bin'))
            self.frame_nums[data_folder] = bin_files.count()
            bin_files_dict[data_folder] = bin_files

        logging.info(self.ratio)
        logging.info(self.frame_nums)
        # Update the frame nums that should be remained
        self.update_frame_nums()
        logging.info(self.target_nums)
        with open(self.log_file, 'a+') as f:
            f.write(f'{time.ctime()}\n')
            f.write(f'ratio: {self.ratio}\n')
            f.write(f'frames before sampling: {self.frame_nums}\n')
            f.write(f'target frames: {self.target_nums}\n')

        for data_folder in os.listdir(self.our_storage().abs_path(src_dir_prefix)):
            bin_files = bin_files_dict[data_folder]
            bin_files = bin_files.sample(False,
                                         self.target_nums[data_folder]
                                         / self.frame_nums[data_folder])

            # copy the bin files to target path
            target_bin_files = bin_files.map(
                lambda elem: self._copy_bin_files(
                    elem,
                    self.src_dir_prefix,
                    self.dst_dir_prefix))
            count = target_bin_files.count()
            logging.info(f'{data_folder} after sampling: {count}')
            with open(self.log_file, 'a+') as f:
                f.write(f'{data_folder} after sampling: {count}\n')

    def update_frame_nums(self):
        self.target_nums = {'LEFT_TURN': 0,
                            'RIGHT_TURN': 0,
                            'U_TURN': 0,
                            'NO_TURN': 0}

        min_frame_num_by_ratio = -1
        for key in self.frame_nums:
            if self.frame_nums[key] == 0:
                del self.target_nums[key]
            else:
                cur_frame_num_by_ratio = self.frame_nums[key] / self.ratio[key]
                if min_frame_num_by_ratio < 0:
                    min_frame_num_by_ratio = cur_frame_num_by_ratio
                else:
                    min_frame_num_by_ratio = min(min_frame_num_by_ratio, cur_frame_num_by_ratio)

        for key in self.target_nums:
            self.target_nums[key] = self.ratio[key] * min_frame_num_by_ratio

    @staticmethod
    def _copy_bin_files(src_file_path, src_dir_prefix, dst_dir_prefix):
        dst_file_path = src_file_path.replace(src_dir_prefix, dst_dir_prefix)
        logging.debug(dst_file_path)
        dst_dir = os.path.dirname(dst_file_path)
        file_utils.makedirs(dst_dir)
        copyfile(src_file_path, dst_file_path)
        return dst_file_path


if __name__ == '__main__':
    DataBalancer().main()
