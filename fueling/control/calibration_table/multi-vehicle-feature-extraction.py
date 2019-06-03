#!/usr/bin/env python
""" extract features for multiple vehicles """
import glob
import os

from absl import flags
import colored_glog as glog
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_utils import pair_cs_pose
from fueling.control.common.sanity_check import sanity_check  # include sanity check
from fueling.control.common.training_conf import inter_result_folder  # intermediate result folder
import fueling.common.bos_client as bos_client
import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.common.time_utils as time_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils


flags.DEFINE_string('input_data_path', 'modules/control/data/records',
                    'Multi-vehicle calibration feature extraction input data path.')


channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
MIN_MSG_PER_SEGMENT = 1
MARKER = 'CompleteCalibrationTable'


def gen_pre_segment(dir_to_msg):
    """Generate new key which contains a segment id part."""
    (vehicle, task_dir), msg = dir_to_msg
    dt = time_utils.msg_time_to_datetime(msg.timestamp)
    segment_id = dt.strftime('%Y%m%d-%H%M')
    return ((vehicle, task_dir, segment_id), msg)


def valid_segment(records):
    msg_segment = spark_helper.cache_and_log(
        'Msgs',
        records
        # PairRDD(vehicle, msg)
        .flatMapValues(record_utils.read_record(channels))
        # PairRDD((vehicle, task_dir, timestamp_per_min), msg)
        .map(gen_pre_segment))
    valid_msg_segment = spark_helper.cache_and_log(
        'valid_segments',
        # PairRDD((vehicle, task_dir, segment_id), msg)
        feature_extraction_rdd_utils.chassis_localization_segment_rdd(
            msg_segment, MIN_MSG_PER_SEGMENT))
    # PairRDD((vehicle, dir_segment, segment_id), msg)
    return spark_op.filter_keys(msg_segment, valid_msg_segment)


def write_h5(elem, origin_prefix, target_prefix):
    (vehicle, segment_dir, segment_id), feature_data = elem
    origin_prefix = os.path.join(origin_prefix, vehicle)
    target_prefix = os.path.join(target_prefix, 'CalibrationTableFeature', vehicle)
    data_set = ((segment_dir, segment_id), feature_data)
    # return feature_data
    return calibration_table_utils.write_h5_train_test(data_set, origin_prefix, target_prefix)


def mark_complete(valid_segment, origin_prefix, target_prefix, MARKER):
    # PairRDD((vehicle, segment_dir, segment_id), msg)
    result_rdd = (
        valid_segment
        # PairRDD(dir_segment, vehicle)
        .map(lambda ((vehicle, segment_dir, segment_id), msgs):
             (segment_dir, vehicle))
        # PariRDD(dir_segment) unique dir
        .distinct()
        # RDD(dir_segment with target_prefix)
        .map(lambda (path, vehicle):
             path.replace(os.path.join(origin_prefix, vehicle),
                          os.path.join(target_prefix, 'CalibrationTableFeature',
                                       vehicle, 'throttle', 'train'), 1))
        # RDD(MARKER files)
        .map(lambda path: os.path.join(path, MARKER)))
    # RDD(dir_MARKER)
    result_rdd.foreach(file_utils.touch)
    return result_rdd


class MultiCalibrationTableFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'multi_calibration_table_feature_extraction')

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated'

        # add sanity check
        if not sanity_check(origin_prefix):
            return

        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_prefix])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, vehicle_type)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        """ get to do jobs """
        """ for run_test only, folder/vehicle/subfolder/*.record.* """
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            origin_vehicle_dir
            # PairRDD(vehicle_type, list_of_records)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*')))
            .mapValues(os.path.dirname))

        """ get conf files """
        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file',
            # PairRDD(vehicle, dir_of_vehicle)
            origin_vehicle_dir
            # PairRDD(vehicle_type, vehicle_conf)
            .mapValues(multi_vehicle_utils.get_vehicle_param))

        self.run(todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix)

    def run_prod(self):
        origin_prefix = self.FLAGS.get('input_data_path')
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        # extract features to intermediate result folder
        target_prefix = os.path.join(inter_result_folder, job_owner, job_id)

        origin_dir = bos_client.abs_path(origin_prefix)

        glog.info("origin_dir: %s" % origin_dir)
        glog.info("target_prefix: %s" % target_prefix)

        # add sanity check
        if not sanity_check(origin_dir):
            return

        """ vehicles """
        vehicles = spark_helper.cache_and_log(
            'conf_file',
            # RDD(input_dir)
            self.to_rdd([origin_dir])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle))
        glog.info("vehicles: %s", vehicles.collect())

        """ get conf files """
        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file',
            # RDD(input_dir)
            self.to_rdd([origin_dir])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle, vehicle)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle, dir_of_vehicle)
            .mapValues(lambda vehicle: os.path.join(origin_dir, vehicle)))

        # PairRDD(vehicle, vehicle_param)
        vehicle_param_conf = vehicle_param_conf.mapValues(multi_vehicle_utils.get_vehicle_param)
        glog.info("vehicle_param_conf: %d", vehicle_param_conf.count())

        # sanity check

        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_dir])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type)))

        """ get to do jobs """
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            # PairRDD(vehicle_type, relative_path_to_vehicle_type)
            origin_vehicle_dir
            # PairRDD(vehicle_type, files)
            .flatMapValues(self.bos().list_files)
            # PairRDD(vehicle_type, 'COMPLETE'_files)
            .filter(lambda key_path: key_path[1].endswith('COMPLETE'))
            # PairRDD(vehicle_type, absolute_path_to_'COMPLETE')
            .mapValues(os.path.dirname))

        processed_dirs = spark_helper.cache_and_log(
            'processed_jobs',
            # PairRDD(vehicle, abs_task_dir)
            todo_task_dirs
            # PairRDD(vehicle, task_dir_with_target_prefix)
            .map(lambda (vehicle, path):
                 (vehicle, path.replace(os.path.join(origin_prefix, vehicle),
                                        os.path.join(target_prefix, 'CalibrationTableFeature',
                                                     vehicle, 'throttle', 'train'), 1)))
            # PairRDD(vehicle, files)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*')))
            # PairRDD(vehicle, file_end_with_MARKER)
            .filter(lambda key_path: key_path[1].endswith(MARKER))
            # PairRDD(vehicle, file_end_with_MARKER with origin prefix)
            .map(lambda (vehicle, path):
                 (vehicle, path.replace(os.path.join(target_prefix, 'CalibrationTableFeature',
                                                     vehicle, 'throttle', 'train'),
                                        os.path.join(origin_prefix, vehicle), 1)))
            # PairRDD(vehicle, dir of file_end_with_MARKER with origin prefix)
            .mapValues(os.path.dirname))

        # PairRDD(vehicle_type, dir_of_todos_with_origin_prefix)
        todo_task_dirs = todo_task_dirs.subtract(processed_dirs)

        self.run(todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix)

    def run(self, todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix):

        records = spark_helper.cache_and_log(
            'Records',
            todo_task_dirs
            # PairRDD(vehicle, files)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*record*')))
            # PairRDD(vehicle, records)
            .filter(lambda (_, end_file): record_utils.is_record_file(end_file))
            # PairRDD(vehicle, (dir, records))
            .mapValues(lambda records: (os.path.dirname(records), records))
            # PairRDD((vehicle, dir), records)
            .map(lambda (vehicle, (record_dir, records)): ((vehicle, record_dir), records)))

        # PairRDD((vehicle, segment_dir, segment_id), msg)
        valid_msg_segments = spark_helper.cache_and_log('Valid_msgs', valid_segment(records))

        parsed_msgs = spark_helper.cache_and_log(
            'parsed_msg',
            # PairRDD((vehicle, dir, segment_id), (chassis_msgs, pose_msgs))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msg_segments)
            # PairRDD((vehicle, dir, segment_id), paired_chassis_msg_pose_msg)
            .mapValues(pair_cs_pose))

        msgs_with_conf = spark_helper.cache_and_log(
            'msgs_with_conf',
            # PairRDD(vehicle, (segment_dir, segment_id, paired_chassis_msg_pose_msg))
            parsed_msgs.map(lambda ((vehicle, segment_dir, segment_id), msgs):
                            (vehicle, (segment_dir, segment_id, msgs)))
            # PairRDD(vehicle,
            #         ((dir_segment, segment_id, paired_chassis_msg_pose_msg), vehicle_param_conf))
            .join(vehicle_param_conf)
            # PairRDD((vehicle, dir_segment, segment_id),
            #         (paired_chassis_msg_pose_msg, vehicle_param_conf))
            .map(lambda (vehicle, ((segment_dir, segment_id, msgs), conf)):
                 ((vehicle, segment_dir, segment_id), (msgs, conf))))

        data_rdd = spark_helper.cache_and_log(
            'data_rdd',
            # PairRDD((vehicle, dir_segment, segment_id),
            #         (paired_chassis_msg_pose_msg, vehicle_param_conf))
            msgs_with_conf
            # PairRDD((vehicle, dir_segment, segment_id), (features, vehicle_param_conf))
            .mapValues(lambda (msgs, conf):
                       (calibration_table_utils.feature_generate(msgs, conf), conf))
            # PairRDD((vehicle, dir_segment, segment_id), (features, vehicle_param_conf))
            .mapValues(lambda (features, conf):
                       (calibration_table_utils.feature_filter(features), conf))
            # PairRDD((vehicle, dir_segment, segment_id), (features, vehicle_param_conf))
            .mapValues(lambda (features, conf):
                       (calibration_table_utils.feature_cut(features, conf), conf))
            # PairRDD((vehicle, dir_segment, segment_id),
            #         ((grid_dict, features), vehicle_param_conf))
            .mapValues(lambda (features, conf):
                       (calibration_table_utils.feature_distribute(features, conf), conf))
            # PairRDD((vehicle, dir_segment, segment_id), feature_data_matrix)
            .mapValues(lambda (features_grid, conf):
                       calibration_table_utils.feature_store(features_grid, conf)))

        # write data to hdf5 files
        result_rdd = spark_helper.cache_and_log(
            'result_rdd',
            # PairRDD((vehicle, dir_segment, segment_id), feature_data_matrix)
            data_rdd
            # RDD(feature_numbers)
            .map(lambda elem: write_h5(elem, origin_prefix, target_prefix)))

        # RDD (dir_segment)
        complete_rdd = spark_helper.cache_and_log(
            'completed_dirs',
            mark_complete(valid_msg_segments, origin_prefix,
                          target_prefix, MARKER))


if __name__ == '__main__':
    MultiCalibrationTableFeatureExtraction().main()
