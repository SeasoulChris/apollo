#!/usr/bin/env python
import copy
import glob
import math
import shutil
import os

from absl import flags
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
from fueling.control.dynamic_model.conf.model_config import feature_extraction
from fueling.control.features.feature_extraction_utils import pair_cs_pose
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.common.time_utils as time_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

# TODO(SHU): redesign proto
flags.DEFINE_string('input_data_path', 'modules/control/data/records',
                    'Multi-vehicle dynamic model input data path.')

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
MIN_MSG_PER_SEGMENT = 100
MARKER = 'CompleteSampleSet'
VEHICLE_CONF = 'vehicle_param.pb.txt'
GEAR = feature_extraction['gear']
INTER_FOLDER = feature_extraction['inter_result_folder']
INCREMENTAL_PROCESS = feature_extraction['incremental_process']


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
        .map(gen_pre_segment), 0)
    valid_msg_segment = spark_helper.cache_and_log(
        'valid_segments',
        # PairRDD((vehicle, task_dir, segment_id), msg)
        feature_extraction_rdd_utils.chassis_localization_segment_rdd(
            msg_segment, MIN_MSG_PER_SEGMENT), 0)
    # PairRDD((vehicle, dir_segment, segment_id), msg)
    return spark_op.filter_keys(msg_segment, valid_msg_segment)


def get_conf_value(msgs):
    # get max value from data
    speed_max = 0.0
    throttle_max = 0.0  # positive value
    brake_max = 0.0  # positive value
    speed_min = 30.0
    acc_min = 0.0  # negative
    acc_max = 0.0
    for msg in msgs:
        chassis, pose_pre = msg
        pose = pose_pre.pose
        heading_angle = pose.heading
        acc_x = pose.linear_acceleration.x
        acc_y = pose.linear_acceleration.y
        acc = acc_x * math.cos(heading_angle) + acc_y * math.sin(heading_angle)
        if GEAR == 1:
            # keep only gear_drive data
            if int(chassis.gear_location) != 1 or chassis.speed_mps < 0:
                continue
        elif GEAR == 2:
            # keep only gear_reverse data
            if int(chassis.gear_location) != 2 or chassis.speed_mps > 0:
                continue
        throttle_max = max(chassis.throttle_percentage, throttle_max)
        brake_max = max(chassis.brake_percentage, brake_max)
        speed_max = max(chassis.speed_mps, speed_max)
        acc_max = max(acc, acc_max)
    return (speed_max, throttle_max, brake_max)


def compare_conf_value(conf_value_x, conf_value_y):
    if not conf_value_x:
        return conf_value_y
    elif not conf_value_y:
        return conf_value_x
    speed_max_x, throttle_max_x, brake_max_x = conf_value_x
    speed_max_y, throttle_max_y, brake_max_y = conf_value_y
    speed_max = max(speed_max_x, speed_max_y)
    throttle_max = max(throttle_max_x, throttle_max_y)
    brake_max = max(brake_max_x, brake_max_y)
    return (speed_max, throttle_max, brake_max)


def write_conf(conf_value, vehicle_param_conf, conf_path, conf):
    cur_conf = copy.deepcopy(conf)
    speed_max, throttle_max, brake_max = conf_value
    cur_conf.speed_max = min(speed_max, conf.speed_max)
    cur_conf.throttle_max = min(throttle_max, conf.throttle_max)
    cur_conf.brake_max = min(brake_max, conf.brake_max)
    logging.info('Load calibration table conf: %s' % cur_conf)
    file_utils.makedirs(conf_path)
    with open(os.path.join(conf_path, 'feature_key_conf.pb.txt'), 'w') as fin:
        fin.write(str(cur_conf))
    return 0


class SampleSet(BasePipeline):

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated'
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        target_prefix = os.path.join(target_prefix, job_owner, job_id)

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
        logging.info('origin_vehicle_dir %s' % origin_vehicle_dir.collect())

        """ get to do jobs """
        """ for run_test only, folder/vehicle/subfolder/*.record.* """
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            origin_vehicle_dir
            # PairRDD(vehicle_type, list_of_records)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*')))
            .mapValues(os.path.dirname))
        logging.info('todo_task_dirs %s' % todo_task_dirs.collect())

        """ get conf files """
        vehicle_param_conf = origin_vehicle_dir

        conf_target_prefix = target_prefix
        logging.info('todo_task_dirs %s' % origin_vehicle_dir.collect())
        logging.info(conf_target_prefix)
        target_param_conf = origin_vehicle_dir.mapValues(
            lambda path: path.replace(origin_prefix, conf_target_prefix, 1))
        logging.info('target_param_conf: %s' % target_param_conf.collect())

        # PairRDD(source_vehicle_param_conf, dest_vehicle_param_conf)
        src_dst_rdd = (origin_vehicle_dir.join(target_param_conf).values().cache())
        # Make dst dirs.
        src_dst_rdd.values().foreach(file_utils.makedirs)
        # Copy confs.
        src_dst_rdd.foreach(lambda src_dst: shutil.copyfile(os.path.join(src_dst[0], VEHICLE_CONF),
                                                            os.path.join(src_dst[1], VEHICLE_CONF)))

        logging.info('todo_task_dirs: %s' % todo_task_dirs.collect())
        logging.info('vehicle_param_conf: %s' % vehicle_param_conf.collect())
        logging.info('origin_prefix: %s' % origin_prefix)
        logging.info('target_prefix: %s' % target_prefix)

        self.run(todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        origin_prefix = self.FLAGS.get('input_data_path')
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')

        # extract features to intermediate result folder
        target_prefix = os.path.join(INTER_FOLDER, job_owner, job_id)
        our_bos = BosClient()
        target_dir = our_bos.abs_path(target_prefix)
        logging.info('target_dir %s' % target_dir)

        # Access partner's storage if provided.
        object_storage = self.partner_object_storage() or our_bos
        origin_dir = object_storage.abs_path(origin_prefix)

        logging.info("origin_dir: %s" % origin_dir)
        logging.info("target_prefix: %s" % target_prefix)

        # TODO(SHU): ADD SANITY check

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
            .mapValues(os.path.dirname)
            .distinct())

        processed_dirs = spark_helper.cache_and_log(
            'processed_jobs',
            # PairRDD(vehicle, abs_task_dir)
            todo_task_dirs
            # PairRDD(vehicle, task_dir_with_target_prefix)
            .mapValues(lambda path: path.replace(origin_prefix, target_prefix, 1))
            # PairRDD(vehicle, files)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*')))
            # PairRDD(vehicle, file_end_with_MARKER)
            .filter(lambda key_path: key_path[1].endswith(MARKER))
            # PairRDD(vehicle, file_end_with_MARKER with origin prefix)
            .mapValues(lambda path: path.replace(origin_prefix, target_prefix, 1))
            # PairRDD(vehicle, dir of file_end_with_MARKER with origin prefix)
            .mapValues(os.path.dirname))

        # PairRDD(vehicle_type, dir_of_todos_with_origin_prefix)
        todo_task_dirs = todo_task_dirs.subtract(processed_dirs)

        self.run(todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix)

    def run(self, todo_task_dirs, vehicle_conf_folder, origin_prefix, target_prefix):
        """ processing RDD """
        # PairRDD(vehicle, vehicle_param)
        vehicle_param_conf = vehicle_conf_folder.mapValues(multi_vehicle_utils.get_vehicle_param)
        logging.info("vehicle_param_conf: %d", vehicle_param_conf.count())

        def _reorg_elements(elements):
            vehicle, record = elements
            record_dir = os.path.dirname(record)
            return ((vehicle, record_dir), record)

        records = (
            # PairRDD(vehicle, vehicle_folder)
            todo_task_dirs
            # PairRDD(vehicle, files)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*record*')))
            # PairRDD(vehicle, record)
            .filter(lambda _end_file: record_utils.is_record_file(_end_file[1]))
            # PairRDD((vehicle, record_dir), record)
            .map(_reorg_elements)
            .cache())

        logging.info('Records %s' % records.collect())

        # PairRDD((vehicle, segment_dir, segment_id), msg)
        valid_msg_segments = valid_segment(records)
        logging.info('Valid_msgs %d' % valid_msg_segments.count())

        parsed_msgs = (
            # PairRDD((vehicle, dir, segment_id), (chassis_msgs, pose_msgs))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msg_segments)
            # PairRDD((vehicle, dir, segment_id), one paired_chassis_msg_pose_msg)
            .flatMapValues(pair_cs_pose)).cache()

        logging.info('parsed_msgs {}'.format(parsed_msgs.first()))

        data_segment_rdd = spark_helper.cache_and_log(
            'get_data_point',
            parsed_msgs
            # PairRDD(vehicle, (dir, timestamp_sec, single data_point))
            .map(feature_extraction_utils.multi_get_data_point), 1)

        def _reorg_elements(elements):
            vehicle, ((data_dir, timestamp_sec, single_data_point), vehicle_param_conf) = elements
            return (vehicle, data_dir, timestamp_sec), (single_data_point, vehicle_param_conf)

        # join data with conf files
        data_segment_rdd = spark_helper.cache_and_log(
            'get_data_point',
            data_segment_rdd
            # PairRDD(vehicle, ((dir, timestamp_sec, single data_point), vehicle_param_conf))
            .join(vehicle_param_conf)
            # PairRDD((vehicle, dir, timestamp_sec), (single data_point, vehicle_param_conf))
            .map(_reorg_elements))

        if GEAR == 1:
            data_segment_rdd = spark_helper.cache_and_log(
                'feature_key_value',
                data_segment_rdd
                # PairRDD((vehicle, dir, feature_key), (timestamp_sec, data_point))
                .map(feature_extraction_utils.multi_gen_feature_key), 1)
        elif GEAR == 2:
            data_segment_rdd = spark_helper.cache_and_log(
                'feature_key_value',
                data_segment_rdd
                # PairRDD((vehicle, dir, feature_key), (timestamp_sec, data_point))
                .map(feature_extraction_utils.multi_gen_feature_key_backwards), 1)

        # count data frames
        logging.info('number of elems: %d' % data_segment_rdd
                     # PairRDD((vehicle, dir, feature_key), (timestamp_sec, data_point) RDD)
                     .groupByKey()
                     # PairRDD((vehicle, dir, feature_key), list of (timestamp_sec, data_point))
                     .mapValues(list).count())

        data_segment_rdd = spark_helper.cache_and_log(
            'gen_segment',
            # PairRDD((vehicle, dir, feature_key), list of (timestamp_sec, data_point))
            data_segment_rdd
            # feature_key != 10000
            .filter(lambda elements: elements[0][2] != 10000)
            # PairRDD((vehicle, dir, feature_key), (timestamp_sec, data_point) RDD)
            .groupByKey()
            # PairRDD((vehicle, dir, feature_key), list of (timestamp_sec, data_point))
            .mapValues(list)
            # # PairRDD((vehicle, dir, feature_key), one segment)
            .flatMapValues(feature_extraction_utils.gen_segment), 0)

        spark_helper.cache_and_log(
            'H5ResultMarkers',
            # PairRDD((vehicle, dir, feature_key), one segment)
            data_segment_rdd
            # PairRDD(vehicle, dir, feature_key), write all segment into a hdf5 file
            .map(lambda elem: feature_extraction_utils.multi_write_segment_with_key(
                elem, origin_prefix, target_prefix))
            # RDD(dir)
            .keys()
            # RDD(dir), which is unique
            .distinct()
            # RDD(MARKER files)
            .map(lambda path: os.path.join(path, MARKER))
            # TODO(SHU): touch path remove job_id and job_owner
            # RDD(MARKER files)
            .map(file_utils.touch))
        return


if __name__ == '__main__':
    SampleSet().main()
