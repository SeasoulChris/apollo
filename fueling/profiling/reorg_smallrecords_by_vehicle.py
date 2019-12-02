#!/usr/bin/env python
import operator
import os
import shutil

import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
import fueling.common.db_backed_utils as db_backed_utils
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.profiling.feature_extraction.multi_job_control_feature_extraction_utils as feature_utils


class ReorgSmallRecords(BasePipeline):
    """Reorg small Records by vehicle as the input data path to control profiling pipeline."""

    def __init__(self):
        self.collection = Mongo().record_collection()

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        records_dir = self.to_rdd(['small-records/2019'])
        origin_prefix = 'small-records/2019'
        target_prefix = 'modules/control/small-records'
        self.run(records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        input_prefix = 'small-records/2019'
        origin_dir = self.our_storage().abs_path(input_prefix)

        target_prefix = 'modules/control/small-records'
        target_dir = self.our_storage().abs_path(target_prefix)

        # TODO: test get vehicles by dir mongo_utils, get empty now

        self.get_vehicles(
            'small-records/2019/2019-11-27/2019-11-27-10-25-44/20191127102544.record')

        self.get_vehicles(
            'small-records/2019/2019-02-25/2019-02-25-16-24-27/20190225163700.record')

        self.get_vehicles(
            'small-records/2019/2019-02-25/2019-02-25-16-24-27')

        exit(1)

        # RDD(small-records dir)
        origin_small_records_dir = spark_helper.cache_and_log(
            'origin_small_records_dir',
            self.to_rdd(self.our_storage().list_files(input_prefix))
            .filter(record_utils.is_record_file)
            .map(lambda record: os.path.dirname(record))
            .distinct()
        )
        logging.info('origin_small_records_dir %s ' %
                     origin_small_records_dir.take(10))

        # processed_dirs = spark_helper.cache_and_log(
        #     'processed_dirs',
        #     self.to_rdd([target_dir])
        #     # RDD([vehicle_type])
        #     .flatMap(multi_vehicle_utils.get_vehicle)
        #     # PairRDD(vehicle_type, [vehicle_type])
        #     .keyBy(lambda vehicle_type: vehicle_type)
        #     # PairRDD(vehicle_type, path_to_vehicle_type)
        #     .mapValues(lambda vehicle_type: os.path.join(target_prefix, vehicle_type))
        #     # PairRDD(vehicle_type, records)
        #     .flatMapValues(self.our_storage().list_files)
        #     # PairRDD(vehicle_type, file endwith REORG_COMPLETE)
        #     .filter(lambda key_path: key_path[1].endswith('REORG_COMPLETE'))
        #     # PairRDD(vehicle_type, path)
        #     .mapValues(os.path.dirname)
        #     .distinct()
        # )
        # logging.info('processed_dirs: %s' % processed_dirs.collect())

        # self.run(origin_small_records_dir, origin_dir, target_dir)

    def run(self, record_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(record_dir)
            record_dir_rdd
            # PairRDD(record_dir, vehicle_name)
            .mapPartitions(self.get_vehicles)
            # RDD(0/1), 1 for success
            .map(lambda dir_map: self.process_dir(
                dir_map[0],
                dir_map[0].replace(origin_prefix,
                                   os.path.join(target_prefix, dir_map[1] + '/'), 1),
                dir_map[1], origin_prefix))
            .cache()
        )

        logging.info(
            'Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    def process_dir(self, record_dir, target_dir, vehicle, origin_dir):
        # record_dir /mnt/bos/small-records/2019/2019-01-03/2019-01-03-15-49-43
        # target_dir /mnt/bos/modules/control/small-records/Mkz7/2019-01-03/2019-01-03-15-49-43

        # create target vehicles directory
        file_utils.makedirs(target_dir)

        # Copy record files
        shutil.copytree(record_dir, target_dir)

        # Copy vehicle_parameter.pb.txt /mnt/bos/modules/control/control_conf/mkz6/vehicle_param.pb.txt
        control_conf_prefix = '/mnt/bos/modules/control/control_conf'
        sorce_conf_dir = os.path.join(
            record_dir.replace(origin_dir, control_conf_prefix, 1), feature_utils.CONF_FILE)
        target_conf_dir = os.path.join(
            control_conf_prefix, vehicle, feature_utils.CONF_FILE)
        shutil.copyfile(sorce_conf_dir, target_conf_dir)

        # Touch Reorg tag
        # TODO(zongbao): need to add filter in multi_job_control_profiling_metrics.py
        file_utils.touch(os.path.join(target_dir, 'REORG_COMPLETE'))

    def get_vehicles(self, record_dirs):
        """Return the (record_dir, vehicle_name) pair"""
        record_dirs = list(record_dirs)
        collection = Mongo().record_collection()
        dir_vehicle_dict = db_backed_utils.lookup_vehicle_for_dirs(
            record_dirs, collection)
        logging.info('dir_vehicle_dict%s' % dir_vehicle_dict)
        dir_vehicle_list = []
        for record_dir, vehicle in dir_vehicle_dict.items():
            # record_dir, Mkz7
            dir_vehicle_list.append((record_dir, vehicle))
        return dir_vehicle_list


if __name__ == '__main__':
    ReorgSmallRecords().main()
