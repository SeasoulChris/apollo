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

VECHILE_PARAM_CON_DIR = 'modules/control/control_conf'
REORG_TAG = 'REORG_COMPLETE'


class ReorgSmallRecordsByVehicle(BasePipeline):
    """Reorg small Records by vehicle as the input data path to control profiling pipeline."""

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        origin_prefix = 'testdata/profiling'
        target_prefix = 'modules/control/small-records'

        # RDD(small-records dir)
        records_dir = spark_helper.cache_and_log(
            'records_dir',
            # RDD(file), start with origin_prefix
            self.to_rdd(self.our_storage().list_files(origin_prefix))
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct()
        )
        # print(self.our_storage().list_files('testdata/control'))
        logging.info(F'records_dir: {records_dir.collect()}')

        self.run(records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        # for testing need check a sub directory to save time
        # input_prefix = 'small-records/2019/2019-06-13'
        input_prefix = 'small-records/2020'
        origin_dir = self.our_storage().abs_path(input_prefix)

        target_prefix = 'modules/control/small-records'
        target_dir = self.our_storage().abs_path(target_prefix)

        # RDD(small-records dir)
        records_dir = spark_helper.cache_and_log(
            'records_dir',
            # RDD(file), start with origin_prefix
            self.to_rdd(self.our_storage().list_files(input_prefix))
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct()
        )

        logging.info(F'records_dir {records_dir.collect()}')

        self.run(records_dir, origin_dir, target_dir)

    def run(self, record_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""

        dir_vehicle_list = self.get_vehicles(record_dir_rdd.collect())
        # dir_vehicle_list[('Transit', '/mnt/bos/small-records/2019/2019-02-25/2019-02-25-16-24-27'),...]
        logging.info(F'dir_vehicle_list{dir_vehicle_list}')

        def _filter_vehicle(vehicle):
            vehicle_parameter_conf_file = os.path.join(
                target_prefix, vehicle, feature_utils.CONF_FILE)
            return not os.path.exists(vehicle_parameter_conf_file)

        # 1. Make vehicle directory and copy parameter config file
        # PairRDD(target_vehicle, source_dir)
        target_vehicle_rdd = spark_helper.cache_and_log(
            'target_vehicle_rdd',
            self.to_rdd(dir_vehicle_list)
            # RDD(vehicle)
            .keys()
            # RDD(vehicle) for case 'mkz7 concord'
            .map(lambda vehicle: vehicle.replace(' ', '_', 1))
            # RDD(vehicle) don't have parameter conf file
            .filter(_filter_vehicle)
        )

        # mkdir for vehicle
        target_vehicle_rdd.foreach(lambda vehicle: file_utils.makedirs(
            os.path.join(target_prefix, vehicle)))
        # Copy vehicle parameter config file
        abs_vehicle_param_path = self.our_storage().abs_path(VECHILE_PARAM_CON_DIR)
        target_vehicle_rdd.foreach(
            lambda vehicle: shutil.copyfile(
                os.path.join(abs_vehicle_param_path, vehicle.lower(), feature_utils.CONF_FILE),
                os.path.join(target_prefix, vehicle, feature_utils.CONF_FILE)
            )
        )

        def _update_rdd(pair_rdd):
            source_dir, target_vehicle_dir = pair_rdd
            # Vehicle at last position
            vehicle = target_vehicle_dir.split('/')[-1]
            # Path replace to target directory with vehicle
            target_vehicle_dir_new = os.path.join(target_prefix, vehicle, source_dir.replace(
                '/mnt/bos/small-records/2020/', '', 1))
            return (source_dir, target_vehicle_dir_new)

        # 2.Copy records
        # PairRDD(source_dir, target_dir)
        dir_vehicle_rdd = spark_helper.cache_and_log(
            'dir_vehicle_rdd',
            self.to_rdd(dir_vehicle_list)
            # PairRDD (vehicle, source_dir) which is not reorgized
            .filter(spark_op.filter_value(lambda target: not os.path.exists(os.path.join(target, REORG_TAG))))
            # PairRDD (source_dir, target_dir)
            .map(spark_op.swap_kv)
            # PairRDD(source_dir, target_dir_with_vehicle)
            # ('/mnt/bos/small-records/2019/2019-02-25/2019-02-25-16-18-12', '/mnt/bos/control/small-records/Transit')]
            .mapValues(lambda vehicle: os.path.join(target_prefix, vehicle))
            # PairRDD(source_dir, target_dir_with_vehicle) sample like:
            # ('/mnt/bos/small-records/2019/2019-02-25/2019-02-25-16-18-12', '/mnt/bos/control/small-records/Transit/2019-02-25/2019-02-25-16-18-12')]
            .map(_update_rdd)
            # PairRDD(source_dir, target_dir_with_vehicle) value unique:
            .filter(spark_op.filter_value(lambda task: not os.path.exists(task)))
        )

        logging.info(F'dir_vehicle_rdd {dir_vehicle_rdd.collect()}')

        # dir_vehicle_rdd.values().foreach(file_utils.makedirs)
        dir_vehicle_rdd.foreach(lambda path: shutil.copytree(path[0], path[1]))

        # 3. Add REORG TAG
        source_dir_rdd = spark_helper.cache_and_log(
            'dir_vehicle_rdd',
            dir_vehicle_rdd
            # RDD source_dir
            .keys()
            # RDD touch flag
            .map(lambda path: file_utils.touch(REORG_TAG)))

        logging.info('reorgize small records by vehicle: All Done, PROD')

    def get_vehicles(self, record_dirs):
        """Return the (record_dir, vehicle_name) pair"""
        logging.info(F'input record_dirs: {record_dirs}')
        collection = Mongo().record_collection()
        dir_vehicle_dict = db_backed_utils.lookup_vehicle_for_dirs(record_dirs, collection)
        logging.info(F'dir_vehicle_dict: {dir_vehicle_dict}')
        dir_vehicle_list = []
        for record_dir, vehicle in dir_vehicle_dict.items():
            dir_vehicle_list.append((vehicle, record_dir))
        return dir_vehicle_list


if __name__ == '__main__':
    ReorgSmallRecordsByVehicle().main()
