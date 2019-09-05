#!/usr/bin/env python
"""
This is a module to run perception benchmark on lidar data
"""

import datetime
import os

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.bos_client as bos_client
import fueling.common.file_utils as file_utils


def write_list_path(src_folder_path, dst_file_path):
    """Write the benchmark input file list to local folder"""
    to_write_items = [os.path.join(src_folder_path, file_path)
                      for file_path in os.listdir(src_folder_path) if os.isfile(file_path)]
    with open(dst_file_path, 'w') as output_file:
        for item in to_write_items:
            output_file.write(item)


def execute_task(source_dir):
    """Execute task by task"""
    glog.info('executing task with src_dir: {}'.format(source_dir))

    # Invoke benchmark binary
    glog.info('start to extract run lidar perception benchmark')

    pcd_file_path = '/apollo/pcd_file_path'
    saved_file_path = '/apollo/saved_file_path'
    label_file_path = '/apollo/label_file_path'
    write_list_path(pcd_file_path, os.path.join(source_dir, 'pcd'))
    write_list_path(saved_file_path, os.path.join(source_dir, 'saved'))
    write_list_path(label_file_path, os.path.join(source_dir, 'label'))

    date_time = datetime.now().strftime("%m-%d-%Y.txt")
    file_utils.makedirs(os.path.join(source_dir, 'results'))
    executable_bin = '/apollo/bazel-bin/modules/perception/tool/benchmark/lidar/lidar_benchmark'
    command = '{} --cloud={} --result={} --groundtruth={} \
               --loading_thread_num=4 --evaluation_thread_num=4 --parallel_processing_num=4 \
               --is_folder {} --reserve="{}" 2>&1 | tee /apollo/{}'.format(
        executable_bin, pcd_file_path, saved_file_path, label_file_path,
        "false", "JACCARD:0.5", date_time)
    glog.info('perception executable command is {}'.format(command))
    return_code = os.system(command)
    glog.info("return code for benchmark is {}".format(return_code))

    cp_command = "cp /apollo/{} {}/results/{}".format(date_time, source_dir, date_time)
    return_code = os.system(cp_command)
    glog.info("return code for copy command is {}".format(return_code))

    if return_code != 0:
        glog.error('failed to execute lidar benchmark for task lidar_benchmark')
        # Print log here only, since rerunning will probably fail again.
        # Need people intervention instead
        return
    glog.info('done executing')


class LidarBenchmarkPipeline(BasePipeline):
    """Extract logsim scenarios from records and save the bag/json pair"""

    def __init__(self):
        """Initialize"""
        BasePipeline.__init__(self, 'perception_benchmark')

    def run_test(self):
        """Local mini test."""
        original_path = '/apollo/benchmark-data'
        self.run(original_path)
        glog.info('Perception Lidar Benchmark: All Done, TEST')

    def run_prod(self):
        """Production."""
        root_dir = bos_client.BOS_MOUNT_PATH
        original_path = '{}/modules/perception/benchmark/testdata'.format(root_dir)
        glog.info('origin path is {}'.format(original_path))
        # RDD(tasks), the tasks without root_dir as prefix
        self.run(original_path)
        glog.info('Perception Lidar Benchmark: All Done, PROD')

    def run(self, original_path):
        """Run the pipeline with given parameters"""
        self.to_rdd([original_path]).foreach(execute_task)


if __name__ == '__main__':
    LidarBenchmarkPipeline().main()
