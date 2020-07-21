#!/usr/bin/env python
"""
This pipeline needs old Ubuntu 14.04 environment. Please run it with
    ./tools/submit-job-to-k8s.sh --image hub.baidubce.com/apollo/spark:ubuntu-14.04_spark-2.4.0 ...
"""
import fnmatch
import os

from absl import flags

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.common.spark_helper as spark_helper
import fueling.common.spark_op as spark_op


BINARY = '/apollo/bazel-bin/modules/data/tools/rosbag_to_record/rosbag_to_record'
MARKER = 'COMPLETE'
PROCESS_LAST_N_DAYS = 20


class BagToRecord(BasePipeline):
    """BagToRecord pipeline."""

    def run_test(self):
        """Run test."""
        # PairRDD(src_bag, dst_record)
        bag_to_record = self.to_rdd([
            ('/apollo/docs/demo_guide/demo_2.0.bag',
             '/fuel/testdata/data/generated/demo_2.0.record'),
        ])
        self.run_internal(bag_to_record)

    def run(self):
        """Run prod."""
        src_prefix = 'stale-rosbags/2020/'
        dst_prefix = 'small-records/2020/'

        storage = self.our_storage()
        # PairRDD(src_dir, src_bag)
        marked_dir_to_bag = spark_op.filter_keys(
            # PairRDD(src_dir, src_bag)
            self.to_rdd(storage.list_files(src_prefix, '.bag')).keyBy(os.path.dirname),
            # RDD(src_dir), which has a MARKER.
            self.to_rdd(storage.list_files(src_prefix, MARKER)).map(os.path.dirname))

        # PairRDD(dst_record, src_bag)
        record_to_bag = (
            # PairRDD(src_dir, src_bag)
            marked_dir_to_bag
            # PairRDD(dst_dir, src_bag)
            .map(spark_op.do_key(lambda src_dir: src_dir.replace(src_prefix, dst_prefix, 1)))
            # PairRDD(dst_bag, src_bag)
            .map(lambda dir_bag:
                 (os.path.join(dir_bag[0], self.bag_name_to_record_name(dir_bag[1])),
                  dir_bag[1])))

        # PairRDD(dst_record, src_bag)
        record_to_bag = spark_op.substract_keys(
            record_to_bag, self.to_rdd(storage.list_files(dst_prefix, '.record'))
        ).filter(spark_op.filter_key(record_utils.filter_last_n_days_records(PROCESS_LAST_N_DAYS)))

        self.run_internal(record_to_bag.map(spark_op.swap_kv))

    def run_internal(self, bag_to_record):
        """Run the pipeline with given arguments."""
        spark_helper.cache_and_log('FinishedJobs',
                                   # PairRDD(src_bag, dst_record)
                                   spark_helper.cache_and_log('SrcBagToDstRecord', bag_to_record)
                                   # RDD(dst_record|None)
                                   .map(self.process_file)
                                   # RDD(dst_record)
                                   .filter(spark_op.not_none)
                                   # RDD(dst_dir)
                                   .map(os.path.dirname)
                                   # RDD(unique_dst_dir)
                                   .distinct()
                                   # RDD(dst_MARKER)
                                   .map(lambda path: os.path.join(path, MARKER))
                                   # RDD(dst_MARKER), which is created
                                   .map(file_utils.touch))

    @staticmethod
    def bag_name_to_record_name(bag_path):
        """Generate record name from bag name."""
        record_name = os.path.basename(bag_path)
        if fnmatch.fnmatch(record_name, '201?-??-??-??-??-??*'):
            record_name = record_name.replace('-', '')[0:14]
        elif record_name.endswith('.bag'):
            record_name = record_name[:-4]
        else:
            record_name = record_name
        return record_name + '.record'

    def process_file(self, bag_to_record):
        """(src_bag, dst_record) -> dst_record if successful else None"""
        bag = bag_to_record[0]
        record = bag_to_record[1]
        file_utils.makedirs(os.path.dirname(record))

        cmd = '"{}" "{}" "{}" --small-channels'.format(BINARY, bag, record)
        ret = os.system(cmd)
        msg = 'SHELL[{}]: {}'.format(ret, cmd)
        if ret != 0:
            logging.error(msg)
            return None
        logging.info(msg)
        return record


if __name__ == '__main__':
    BagToRecord().main()
