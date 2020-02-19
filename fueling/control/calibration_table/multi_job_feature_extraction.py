#!/usr/bin/env python
""" extract features for multiple vehicles """
from datetime import datetime
import glob
import shutil
import os

import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.common.base_pipeline import BasePipeline
from fueling.common.partners import partners
from fueling.control.features.feature_extraction_utils import pair_cs_pose
from fueling.control.common.sanity_check import sanity_check  # include sanity check
from fueling.control.common.training_conf import inter_result_folder  # intermediate result folder
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.common.redis_utils as redis_utils
import fueling.common.time_utils as time_utils
import fueling.control.common.multi_job_utils as multi_job_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
# import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
MIN_MSG_PER_SEGMENT = 1
MARKER = 'CompleteCalibrationTable'
VEHICLE_CONF = 'vehicle_param.pb.txt'


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
    target_prefix = os.path.join(target_prefix, vehicle)
    data_set = ((segment_dir, segment_id), feature_data)
    # return feature_data
    return calibration_table_utils.write_h5_train_test(data_set, origin_prefix, target_prefix)


def mark_complete(valid_segment, origin_prefix, target_prefix, MARKER):
   # PairRDD((vehicle, segment_dir, segment_id), msg)
    result_rdd = (
        valid_segment
        # RDD((vehicle, segment_dir, segment_id))
        .keys()
        # RDD(segment_dir)
        .map(lambda elements: elements[1])
        # RDD(segment_dir), which is unique.
        .distinct()
        # RDD(segment_dir), with target_prefix.
        .map(lambda path: path.replace(origin_prefix, target_prefix, 1))
        # RDD(throttle_train_dir_marker)
        .map(lambda path: os.path.join(path, 'throttle/train', MARKER)))
    # RDD(dir_MARKER)
    result_rdd.foreach(file_utils.touch)
    return result_rdd


class MultiJobFeatureExtraction(BasePipeline):

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated'
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')

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
        vehicle_param_conf = origin_vehicle_dir

        conf_target_prefix = target_prefix
        logging.info('todo_task_dirs %s' % origin_vehicle_dir.collect())
        logging.info(conf_target_prefix)
        target_param_conf = origin_vehicle_dir.mapValues(
            lambda path: path.replace(origin_prefix, conf_target_prefix, 1))
        logging.info('target_param_conf: %s' % target_param_conf.collect())
        print("origin_vehicle_dir.join", origin_vehicle_dir.join(target_param_conf).collect())

        # PairRDD(source_vehicle_param_conf, dest_vehicle_param_conf))
        src_dst_rdd = origin_vehicle_dir.join(target_param_conf).values().cache()
        # Create dst dirs and copy conf file to them.
        src_dst_rdd.values().foreach(file_utils.makedirs)
        src_dst_rdd.foreach(lambda src_dst: shutil.copyfile(os.path.join(src_dst[0], VEHICLE_CONF),
                                                            os.path.join(src_dst[1], VEHICLE_CONF)))

        self.run(todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix)

    def run_prod(self):
        origin_prefix = self.FLAGS.get('input_data_path', 'modules/control/data/records')
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')

        # extract features to intermediate result folder
        target_prefix = os.path.join(inter_result_folder, job_owner, job_id)
        our_storage = self.our_storage()
        target_dir = our_storage.abs_path(target_prefix)
        logging.info('target_dir %s' % target_dir)

        # Access partner's storage if provided.
        object_storage = self.partner_storage() or our_storage
        origin_dir = object_storage.abs_path(origin_prefix)

        logging.info("origin_dir: %s" % origin_dir)
        logging.info("target_prefix: %s" % target_prefix)

        job_type, job_size = 'VEHICLE_CALIBRATION', file_utils.getDirSize(origin_dir)
        redis_key = F'External_Partner_Job.{job_owner}.{job_type}.{job_id}'
        redis_value = {'begin_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                       'job_size': job_size,
                       'job_status': 'running'}
        redis_utils.redis_extend_dict(redis_key, redis_value)

        # Add sanity check
        partner = partners.get(job_owner)
        email_receivers = email_utils.CONTROL_TEAM + email_utils.DATA_TEAM
        if partner:
            email_receivers.append(partner.email)
        if not sanity_check(origin_dir, job_owner, job_id, email_receivers):
            return

        """ vehicles """
        vehicles = spark_helper.cache_and_log(
            'conf_file',
            # RDD(input_dir)
            self.to_rdd([origin_dir])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle))
        logging.info("vehicles: %s", vehicles.collect())

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

        # copy vehicle param configure file to target folder
        # target dir
        # PairRDD (vehicle, dst_path)
        target_param_conf = vehicle_param_conf.mapValues(
            lambda path: path.replace(origin_dir, target_dir, 1))
        # PairRDD (src_path, dst_path)
        src_dst_rdd = vehicle_param_conf.join(target_param_conf).values().cache()

        src_dst_rdd.values().foreach(file_utils.makedirs)
        src_dst_rdd.foreach(lambda src_dst: shutil.copyfile(os.path.join(src_dst[0], VEHICLE_CONF),
                                                            os.path.join(src_dst[1], VEHICLE_CONF)))

        logging.info('copy vehicle param conf from src to dst: %s' % src_dst_rdd.collect())

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
            .flatMapValues(object_storage.list_files))

        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            todo_task_dirs
            # TODO: find a better way to get the folder dirs
            # PairRDD(vehicle_type, 'COMPLETE'_files)
            # .filter(lambda key_path: key_path[1].endswith('COMPLETE'))
            # PairRDD(vehicle_type, absolute_path_to_records)
            .mapValues(os.path.dirname)
            .distinct())
        # print('todo_task_dirs: ', todo_task_dirs2.collect())
        # return

        processed_dirs = spark_helper.cache_and_log(
            'processed_jobs',
            # PairRDD(vehicle, abs_task_dir)
            todo_task_dirs
            # PairRDD(vehicle, task_dir_with_target_prefix)
            .mapValues(lambda path: path.replace(origin_prefix, target_prefix, 1))
            # PairRDD(vehicle, train_task_dir_with_target_prefix)
            .mapValues(lambda path: os.path.join(path, 'throttle/train'))
            # PairRDD(vehicle, files)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*')))
            # PairRDD(vehicle, file_end_with_MARKER)
            .filter(lambda key_path: key_path[1].endswith(MARKER))
            # PairRDD(vehicle, file_end_with_MARKER with origin prefix)
            .mapValues(lambda path: path.replace(target_prefix, origin_prefix, 1))
            .mapValues(lambda path: path.replace('throttle/train/', '', 1))
            # PairRDD(vehicle, dir of file_end_with_MARKER with origin prefix)
            .mapValues(os.path.dirname))

        # PairRDD(vehicle_type, dir_of_todos_with_origin_prefix)
        todo_task_dirs = todo_task_dirs.subtract(processed_dirs)

        self.run(todo_task_dirs, vehicle_param_conf, origin_dir, target_dir)

    def run(self, todo_task_dirs, vehicle_conf_folder, origin_prefix, target_prefix):

        # PairRDD(vehicle, vehicle_param)
        vehicle_param_conf = vehicle_conf_folder.mapValues(multi_vehicle_utils.get_vehicle_param)
        logging.info("vehicle_param_conf: %d", vehicle_param_conf.count())

        def _add_dir_to_key(vehicle_and_record):
            vehicle, record = vehicle_and_record
            record_dir = os.path.dirname(record)
            return (vehicle, record_dir), record

        records = spark_helper.cache_and_log(
            'Records',
            todo_task_dirs
            # PairRDD(vehicle, file)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*record*')))
            # PairRDD(vehicle, record)
            .filter(spark_op.filter_value(record_utils.is_record_file))
            # PairRDD((vehicle, dir), record)
            .map(_add_dir_to_key))

        # PairRDD((vehicle, segment_dir, segment_id), msg)
        valid_msg_segments = valid_segment(records)
        logging.info('Valid_msgs %d' % valid_msg_segments.count())

        parsed_msgs = (
            # PairRDD((vehicle, dir, segment_id), (chassis_msgs, pose_msgs))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msg_segments)
            # PairRDD((vehicle, dir, segment_id), paired_chassis_msg_pose_msg)
            .mapValues(pair_cs_pose)).cache()

        logging.info('parsed_msgs %d' % parsed_msgs.count())

        # update conf file for each vehicle
        vehicle_msgs_rdd = (
            parsed_msgs
            # PairRDD(vehicle, paired_chassis_msg_pose_msg)
            .map(spark_op.do_key(lambda key: key[0]))
            # PairRDD(vehicle, (speed_min, speed_max, throttle_max, brake_max))
            .mapValues(multi_job_utils.get_conf_value)
            # PairRDD(vehicle, (speed_min, speed_max, throttle_max, brake_max))
            .reduceByKey(multi_job_utils.compare_conf_value))
        logging.info("vehicle_msgs_rdd: %s" % str(vehicle_msgs_rdd.collect()))

        def _write_conf(vehicle_confs):
            vehicle, (conf_value, conf) = vehicle_confs
            multi_job_utils.write_conf(conf_value, conf, os.path.join(target_prefix, vehicle))

        # write conf value to calibratin table training conf files
        write_conf_rdd = (
            vehicle_msgs_rdd
            # PairRDD(vehicle, ((speed_min, speed_max, throttle_max, brake_max), conf))
            .join(vehicle_param_conf)
            # RDD(0)
            .map(_write_conf))

        logging.info('target_prefix: %s' % target_prefix)
        logging.info('vehicle_msgs_rdd: % d' % write_conf_rdd.count())

        # get train conf files
        #
        train_conf = (
            vehicle_conf_folder
            # PairRDD(vehicle, target_vehicle_folder)
            .mapValues(lambda path: path.replace(origin_prefix, target_prefix, 1))
            # PairRDD(vehicle, 0)
            .mapValues(multi_job_utils.get_train_conf))

        logging.info('train_conf_files %s' % train_conf.collect())

        conf = spark_helper.cache_and_log(
            'conf',
            # PairRDD(vehicle, (vehicle_conf, train_conf))
            vehicle_param_conf.join(train_conf))

        def _reorg_parsed_msgs(elements):
            (vehicle, segment_dir, segment_id), msgs = elements
            return vehicle, (segment_dir, segment_id, msgs)

        msgs_with_conf = spark_helper.cache_and_log(
            'msgs_with_conf',
            # PairRDD(vehicle, (segment_dir, segment_id, paired_chassis_msg_pose_msg))
            parsed_msgs.map(_reorg_parsed_msgs)
            # PairRDD(vehicle,
            #         ((dir_segment, segment_id, paired_chassis_msg_pose_msg),
            #          (vehicle_conf, train_conf)))
            .join(conf), 0)

        elem = msgs_with_conf.first()
        # vehicle_conf
        print('vehicle_conf ', elem[1][1][0])
        # train_conf
        print('train_conf ', elem[1][1][1])

        def _reorg_msgs_with_conf(elements):
            vehicle, ((segment_dir, segment_id, msgs), (vehicle_conf, train_conf)) = elements
            return (vehicle, segment_dir, segment_id), (msgs, vehicle_conf, train_conf)

        # PairRDD((vehicle, dir_segment, segment_id),
        #         (paired_chassis_msg_pose_msg, vehicle_param_conf))
        msgs_with_conf = msgs_with_conf.map(_reorg_msgs_with_conf)
        logging.info('msgs_with_conf: %d' % msgs_with_conf.count())

        data_rdd = spark_helper.cache_and_log(
            'data_rdd',
            # PairRDD((vehicle, dir_segment, segment_id),
            #         (paired_chassis_msg_pose_msg, vehicle_param_conf))
            msgs_with_conf
            # PairRDD((vehicle, dir_segment, segment_id), (features, vehicle_param_conf))
            .mapValues(lambda input3: multi_job_utils.gen_data(input3[0], input3[1], input3[2])),
            0)

        # write data to hdf5 files
        spark_helper.cache_and_log(
            'result_rdd',
            # PairRDD((vehicle, dir_segment, segment_id), feature_data_matrix)
            data_rdd
            # RDD(feature_numbers)
            .map(lambda elem: write_h5(elem, origin_prefix, target_prefix)))

        # RDD (dir_segment)
        spark_helper.cache_and_log(
            'completed_dirs',
            mark_complete(valid_msg_segments, origin_prefix, target_prefix, MARKER))


if __name__ == '__main__':
    MultiJobFeatureExtraction().main()
