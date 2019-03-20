from collections import Counter
import operator
import os

import pyspark_utils.op as spark_op

import fueling.common.record_utils as record_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

def record_to_msgs_rdd(dir_to_records_rdd, root_dir, WANTED_VEHICLE, 
                       channels, MIN_MSG_PER_SEGMENT):
    # PairRDD(absolute_dir, absolute_path_record)
    dir_to_records = absolute_path_rdd(dir_to_records_rdd, root_dir).cache()
    # RDD(aboslute_dir) which include records of the wanted vehicle
    selected_vehicles = wanted_vehicle_rdd(dir_to_records, WANTED_VEHICLE)
    # PairRDD((dir, timestamp_per_min), msg)
    dir_to_msgs = msg_rdd(dir_to_records, selected_vehicles, channels).cache()
    # RDD(dir, timestamp_per_min)
    valid_segments = chassis_localization_segment_rdd(dir_to_msgs, MIN_MSG_PER_SEGMENT)
    # PairRDD((dir_segment, segment_id), (chassis_list, pose_list))
    parsed_msgs = chassis_localization_parsed_msg_rdd(dir_to_msgs, valid_segments)
    return parsed_msgs

def absolute_path_rdd(dir_to_records_rdd,root_dir):
    return (
        # PairRDD(dir, record)
        dir_to_records_rdd
        # PairRDD(absolute_dir, absolute_path_record)
        .map(lambda x: (os.path.join(root_dir, x[0]), os.path.join(root_dir, x[1]))))

def wanted_vehicle_rdd(dir_to_records, WANTED_VEHICLE):
    return (
        # PairRDD(aboslute_dir, vehicle_type)
        feature_extraction_utils.get_vehicle_of_dirs(dir_to_records)
        # PairRDD(aboslute_dir, wanted_vehicle_type)
        .filter(spark_op.filter_value(lambda vehicle: vehicle == WANTED_VEHICLE))
        # RDD(aboslute_dir) which include records of the wanted vehicle
        .keys())

def msg_rdd(dir_to_records, selected_vehicles, channels):
    return (
        # PairRDD(aboslute_path_dir, aboslute_path_record)
        # which include records of the wanted vehicle
        spark_op.filter_keys(dir_to_records, selected_vehicles)
        # PairRDD(dir, msg)
        .flatMapValues(record_utils.read_record(channels))
        # PairRDD((dir, timestamp_per_min), msg)
        .map(feature_extraction_utils.gen_pre_segment))

# for CHASSIS and LOCALIZATION
def chassis_localization_segment_rdd(dir_to_msgs, MIN_MSG_PER_SEGMENT):
    return (
        dir_to_msgs
        # PairRDD((dir, timestamp_per_min), topic_counter)
        .mapValues(lambda msg: Counter([msg.topic]))
        # PairRDD((dir, timestamp_per_min), topic_counter)
        .reduceByKey(operator.add)
        # PairRDD((dir, timestamp_per_min), topic_counter)
        .filter(spark_op.filter_value(
            lambda counter:
                counter.get(record_utils.CHASSIS_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT and
                counter.get(record_utils.LOCALIZATION_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT))
        # RDD((dir, timestamp_per_min))
        .keys())

def chassis_localization_parsed_msg_rdd(dir_to_msgs, valid_segments):
    return (
        # PairRDD((dir_segment, segment_id), msg) to valid segments
        spark_op.filter_keys(dir_to_msgs, valid_segments)
        # PairRDD((dir_segment, segment_id), msgs)
        .groupByKey()
        # PairRDD((dir_segment, segment_id), proto_dict)
        .mapValues(record_utils.messages_to_proto_dict())
        # PairRDD((dir_segment, segment_id), (chassis_list, pose_list))
        .mapValues(lambda proto_dict: (proto_dict[record_utils.CHASSIS_CHANNEL],
                                       proto_dict[record_utils.LOCALIZATION_CHANNEL])))
