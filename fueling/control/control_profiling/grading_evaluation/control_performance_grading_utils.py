""" Control performance grading related utils. """
#!/usr/bin/env python

import math

import pyspark_utils.op as spark_op

import common.proto_utils as proto_utils
import fueling.common.colored_glog as glog
import modules.data.fuel.fueling.control.proto.control_profiling_pb2 \
    as control_profiling_conf
from fueling.control.control_profiling.conf.control_channel_conf import FEATURE_INDEX

FILENAME_CONTROL_PROFILING = \
    "/apollo/modules/data/fuel/fueling/control/conf/control_profiling_conf.pb.txt"
CONTROL_PROFILING = control_profiling_conf.ControlProfiling()
proto_utils.get_pb_from_text_file(FILENAME_CONTROL_PROFILING, CONTROL_PROFILING)

# Define all the constant variables
IDX = FEATURE_INDEX

ACCELERATION_HARSH_LIMIT = CONTROL_PROFILING.control_metrics.acceleration_harsh_limit
CURVATURE_HARSH_LIMIT = CONTROL_PROFILING.control_metrics.curvature_harsh_limit

SPEED_STILL = CONTROL_PROFILING.control_metrics.speed_still
HEADING_STILL = CONTROL_PROFILING.control_metrics.heading_still
CURVATURE_STILL = CONTROL_PROFILING.control_metrics.curvature_still

STATION_ERR_THOLD = CONTROL_PROFILING.control_metrics.station_error_thold
SPEED_ERR_THOLD = CONTROL_PROFILING.control_metrics.speed_error_thold
LATERAL_ERR_THOLD = CONTROL_PROFILING.control_metrics.lateral_error_thold
LATERAL_ERR_RATE_THOLD = CONTROL_PROFILING.control_metrics.lateral_error_rate_thold
HEADING_ERR_THOLD = CONTROL_PROFILING.control_metrics.heading_error_thold
HEADING_ERR_RATE_THOLD = CONTROL_PROFILING.control_metrics.heading_error_rate_thold

LON_ACCELERATION_THOLD = CONTROL_PROFILING.control_metrics.lon_acceleration_thold

CONTROL_PERIOD = CONTROL_PROFILING.control_period
CONTROL_FRAME_NUM = CONTROL_PROFILING.control_frame_num
CONTROL_COMMAND_PCT = CONTROL_PROFILING.control_command_pct
MIN_SAMPLE_SIZE = CONTROL_PROFILING.min_sample_size


def filter_rdd_value(prefilter_rdd=None, value_idx=0, lowerlimit=0, upperlimit=float('inf')):
    """Filter rdd data with respect to the selected variable"""
    filtered_key = (
        # PairRDD((dir, timestamp_sec), list of the selected variables)
        prefilter_rdd
        # PairRDD((dir, timestamp_sec), list of the selected variables)
        .filter(spark_op.filter_value(lambda value:
                                      abs(value[value_idx]) >= lowerlimit and
                                      abs(value[value_idx]) <= upperlimit))
        # RDD((dir, timestamp_sec))
        .keys())
    # RDD((dir, timestamp_sec))
    return filtered_key

def compute_rdd_std(precompute_rdd=None, elem_num=0):
    """Compute the customized stardard deviation of the rdd value"""
    std_rdd = (
        # PairRDD((dir, timestamp_sec), selected variables)
        precompute_rdd
        # RDD(Square of variable)
        .map(lambda value: value **2)
        # Normalized STD
        .sum() / (elem_num -1) **0.5)
        # Normalized STD
    return std_rdd

def compute_rdd_peak(precompute_rdd=None, value_idx=0, threshold=float('inf'), title=""):
    """Compute the peak value of the selected rdd value"""
    elem_num = precompute_rdd.count()
    peak_rdd = (
        # PairRDD((dir, timestamp_sec), list of the selected variables)
        precompute_rdd
        # RDD(selected variable)
        .map(lambda value: abs(value[1][value_idx]))
        # Normalized maximum
        .max() / threshold)
    # Tuple (grading items, grading value, sample size)
    return (title, peak_rdd, elem_num)

def count_rdd_beyond(precompute_rdd=None, value_idx=0, threshold=float('inf'), title=""):
    """Count the total number of the selected rdd value which is beyond the threshold"""
    elem_num = precompute_rdd.count()
    count_rdd = (
        # RDD((dir, timestamp_sec))
        filter_rdd_value(precompute_rdd, value_idx, threshold)
        # Percentage of selected keys from the total keys
        .count() / elem_num)
    # Tuple (grading items, grading value, sample size)
    return (title, count_rdd, elem_num)


def compute_rdd_control_usage(
        precompute_rdd=None, compute_value_idx=0, THOLD_value_idx=0, threshold=0, title=""):
    """Compute the overal percentage of the control command usage with the std format"""
    if THOLD_value_idx is None:
        # RDD((dir, timestamp_sec))
        key_harsh = precompute_rdd.keys()
    else:
        # RDD((dir, timestamp_sec))
        key_harsh = filter_rdd_value(precompute_rdd, THOLD_value_idx, threshold)
    elem_num = key_harsh.count()
    if elem_num < MIN_SAMPLE_SIZE:
        # Tuple (grading items, grading value, sample size)
        return (title, 0.00, elem_num)
    processed_rdd = (
        # PairRDD((dir, timestamp_sec), list of the selected variables)
        spark_op.filter_keys(precompute_rdd, key_harsh)
        # RDD(normalized variable)
        .map(lambda value: value[1][compute_value_idx] / CONTROL_COMMAND_PCT))
    # Tuple (grading items, grading value, sample size)
    return (title, compute_rdd_std(processed_rdd, elem_num), elem_num)


def performance_grading(data_rdd):
    """ genearate the score table to grade the overall control performance """

    # TODO(Yu): generate the uniform function to process stardard-deviation
    # and stardard-deviation under "harsh" sernarios

    # Tuple list [(grading items, grading value, sample size)]
    grading_results = [("Grading Items", "Grading Values", "Sampling Size")]

    # Compute station error stardard-deviation, in percentage of desired station
    # movement in unit time
    elem_num = data_rdd.count()
    processed_rdd = (
        # RDD(normalized variable)
        data_rdd.map(lambda value:
                     value[1][IDX["station_error"]] /
                     (max(value[1][IDX["speed_reference"]], SPEED_STILL) *
                      CONTROL_PERIOD *CONTROL_FRAME_NUM)))
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(("lon_station_err_std",
                            compute_rdd_std(processed_rdd, elem_num), elem_num))

    # Compute station error stardard-deviation under harsh accleration, in
    # percentage of desired station movement in unit time
    # RDD((dir, timestamp_sec))
    key_harsh = filter_rdd_value(data_rdd, IDX["acceleration_reference"],
                                 ACCELERATION_HARSH_LIMIT)
    elem_num = key_harsh.count()
    if elem_num >= MIN_SAMPLE_SIZE:
        processed_rdd = (
            # PairRDD((dir, timestamp_sec), list of the selected variables)
            spark_op.filter_keys(data_rdd, key_harsh)
            # RDD(normalized variable)
            .map(lambda value:
                 value[1][IDX["station_error"]] /
                 (max(value[1][IDX["speed_reference"]], SPEED_STILL) *
                  CONTROL_PERIOD *CONTROL_FRAME_NUM)))
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lon_station_err_std_harsh",
                                compute_rdd_std(processed_rdd, elem_num), elem_num))
    else:
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lon_station_err_std_harsh", 0.00, elem_num))


    # Compute speed error stardard-deviation, in percentage of desired speed
    elem_num = data_rdd.count()
    processed_rdd = (
        # RDD(normalized variable)
        data_rdd.map(lambda value: value[1][IDX["speed_error"]] /
                     (max(value[1][IDX["speed_reference"]], SPEED_STILL))))
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(("lon_speed_err_std",
                            compute_rdd_std(processed_rdd, elem_num), elem_num))

    # Compute speed error stardard-deviation under harsh accleration, in
    # percentage of desired speed
    # RDD((dir, timestamp_sec))
    key_harsh = filter_rdd_value(data_rdd, IDX["acceleration_reference"], ACCELERATION_HARSH_LIMIT)
    elem_num = key_harsh.count()
    if elem_num >= MIN_SAMPLE_SIZE:
        processed_rdd = (
            # PairRDD((dir, timestamp_sec), list of the selected variables)
            spark_op.filter_keys(data_rdd, key_harsh)
            # RDD(normalized variable)
            .map(lambda value: value[1][IDX["speed_error"]] /
                 (max(value[1][IDX["speed_reference"]], SPEED_STILL))))
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lon_speed_err_std_harsh",
                                compute_rdd_std(processed_rdd, elem_num), elem_num))
    else:
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lon_speed_err_std_harsh", 0.00, elem_num))


    # Compute lateral error stardard-deviation, in percentage of desired lateral
    # movement in unit time
    elem_num = data_rdd.count()
    processed_rdd = (
        # RDD(normalized variable)
        data_rdd.map(lambda value:
                     value[1][IDX["lateral_error"]] /
                     (max(value[1][IDX["heading_reference"]], HEADING_STILL) *
                      value[1][IDX["linear_velocity"]] *CONTROL_PERIOD *CONTROL_FRAME_NUM)))
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(("lateral_err_std", compute_rdd_std(processed_rdd, elem_num), elem_num))

    # Compute lateral stardard-deviation under harsh accleration, in percentage
    # of desired lateral movement in unit time
    # RDD((dir, timestamp_sec))
    key_harsh = filter_rdd_value(data_rdd, IDX["curvature_reference"], CURVATURE_HARSH_LIMIT)
    elem_num = key_harsh.count()
    if elem_num >= MIN_SAMPLE_SIZE:
        processed_rdd = (
            # PairRDD((dir, timestamp_sec), list of the selected variables)
            spark_op.filter_keys(data_rdd, key_harsh)
            # RDD(normalized variable)
            .map(lambda value:
                 value[1][IDX["lateral_error"]] /
                 (max(value[1][IDX["heading_reference"]], HEADING_STILL) *
                  value[1][IDX["linear_velocity"]] *CONTROL_PERIOD *CONTROL_FRAME_NUM)))
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lateral_err_std_harsh",
                                compute_rdd_std(processed_rdd, elem_num), elem_num))
    else:
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lateral_err_std_harsh", 0.00, elem_num))


    # Compute lateral error rate stardard-deviation, in percentage of desired lateral rate
    elem_num = data_rdd.count()
    processed_rdd = (
        # RDD((dir, timestamp_sec))
        data_rdd.map(lambda value:
                     value[1][IDX["lateral_error_rate"]] /
                     (max(value[1][IDX["heading_reference"]], HEADING_STILL) *
                      value[1][IDX["linear_velocity"]])))
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(("lateral_err_rate_std",
                            compute_rdd_std(processed_rdd, elem_num), elem_num))

    # Compute lateral rate stardard-deviation under harsh accleration, in
    # percentage of desired lateral rate
    # RDD((dir, timestamp_sec))
    key_harsh = filter_rdd_value(data_rdd, IDX["curvature_reference"], CURVATURE_HARSH_LIMIT)
    elem_num = key_harsh.count()
    if elem_num >= MIN_SAMPLE_SIZE:
        processed_rdd = (
            # PairRDD((dir, timestamp_sec), list of the selected variables)
            spark_op.filter_keys(data_rdd, key_harsh)
            # RDD(normalized variable)
            .map(lambda value: value[1][IDX["lateral_error_rate"]] /
                 (max(value[1][IDX["heading_reference"]], HEADING_STILL) *
                  value[1][IDX["linear_velocity"]])))
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lateral_err_rate_std_harsh",
                                compute_rdd_std(processed_rdd, elem_num), elem_num))
    else:
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("lateral_err_rate_std_harsh", 0.00, elem_num))


    # Compute heading error stardard-deviation, in percentage of desired heading
    elem_num = data_rdd.count()
    processed_rdd = (
        # RDD(normalized variable)
        data_rdd.map(lambda value: value[1][IDX["heading_error"]] /
                     (max(value[1][IDX["heading_reference"]], HEADING_STILL))))
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(("heading_err_std", compute_rdd_std(processed_rdd, elem_num), elem_num))

    # Compute heading error stardard-deviation under harsh accleration, in
    # percentage of desired heading
    # RDD((dir, timestamp_sec))
    key_harsh = filter_rdd_value(data_rdd, IDX["curvature_reference"], CURVATURE_HARSH_LIMIT)
    elem_num = key_harsh.count()
    if elem_num >= MIN_SAMPLE_SIZE:
        processed_rdd = (
            # PairRDD((dir, timestamp_sec), list of the selected variables)
            spark_op.filter_keys(data_rdd, key_harsh)
            # RDD(normalized variable)
            .map(lambda value: value[1][IDX["heading_error"]] /
                 (max(value[1][IDX["heading_reference"]], HEADING_STILL))))
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("heading_err_std_harsh",
                                compute_rdd_std(processed_rdd, elem_num), elem_num))
    else:
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("heading_err_std_harsh", 0.00, elem_num))


    # Compute heading error rate stardard-deviation, in percentage of desired heading rate
    elem_num = data_rdd.count()
    processed_rdd = (
        # RDD(normalized variable)
        data_rdd.map(lambda value:
                     value[1][IDX["heading_error_rate"]] /
                     (max(value[1][IDX["curvature_reference"]], CURVATURE_STILL) *
                      value[1][IDX["linear_velocity"]])))
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(("heading_err_rate_std",
                            compute_rdd_std(processed_rdd, elem_num), elem_num))

    # Compute heading rate stardard-deviation under harsh accleration, in
    # percentage of desired heading rate
    # RDD((dir, timestamp_sec))
    key_harsh = filter_rdd_value(data_rdd, IDX["curvature_reference"], CURVATURE_HARSH_LIMIT)
    elem_num = key_harsh.count()
    if elem_num >= MIN_SAMPLE_SIZE:
        processed_rdd = (
            # PairRDD((dir, timestamp_sec), list of the selected variables)
            spark_op.filter_keys(data_rdd, key_harsh)
            # RDD(normalized variable)
            .map(lambda value:
                 value[1][IDX["heading_error_rate"]] /
                 (max(value[1][IDX["curvature_reference"]], CURVATURE_STILL) *
                  value[1][IDX["linear_velocity"]])))
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("heading_err_rate_std_harsh",
                                compute_rdd_std(processed_rdd, elem_num), elem_num))
    else:
        # Tuple list [(grading items, grading value, sample size)]
        grading_results.append(("heading_err_rate_std_harsh", 0.00, elem_num))


    # Compute station error peak value, in percentage of station error threshold
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_peak(data_rdd, IDX["station_error"],
                                            STATION_ERR_THOLD, "station_err_peak"))
    # Compute speed error peak value, in percentage of speed error threshold
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_peak(data_rdd, IDX["speed_error"],
                                            SPEED_ERR_THOLD, "speed_err_peak"))
    # Compute lateral error peak value, in percentage of lateral error threshold
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_peak(data_rdd, IDX["lateral_error"],
                                            LATERAL_ERR_THOLD, "lateral_err_peak"))
    # Compute lateral error rate peak value, in percentage of lateral error rate threshold
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_peak(data_rdd, IDX["lateral_error_rate"],
                                            LATERAL_ERR_RATE_THOLD, "lateral_err_rate_peak"))
    # Compute heading error peak value, in percentage of heading error threshold
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_peak(data_rdd, IDX["heading_error"],
                                            HEADING_ERR_THOLD, "heading_err_peak"))
    # Compute heading error rate peak value, in percentage of heading error rate threshold
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_peak(data_rdd, IDX["heading_error_rate"],
                                            HEADING_ERR_RATE_THOLD, "heading_err_rate_peak"))


    # Count the bad-sensation frames due to longitudinal acceleration ,
    # in percentage of total frame number
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(count_rdd_beyond(data_rdd, IDX["acceleration_cmd"],
                                            LON_ACCELERATION_THOLD, "lon_acc_bad_sensation"))

    # TODO(Yu): generate the "jerk" data for sensation analysis

    # Compute longitudinal throttle control usage
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_control_usage(data_rdd, IDX["throttle_cmd"], None, None,
                                                     "throttle_control_usage"))
    # Compute longitudinal throttle control usage under harsh acceleration
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_control_usage(data_rdd, IDX["throttle_cmd"],
                                                     IDX["acceleration_reference"],
                                                     ACCELERATION_HARSH_LIMIT,
                                                     "throttle_control_usage_harsh"))
    # Compute longitudinal brake control usage
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_control_usage(data_rdd, IDX["brake_cmd"], None, None,
                                                     "brake_control_usage"))
    # Compute longitudinal brake control usage under harsh acceleration
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_control_usage(data_rdd, IDX["brake_cmd"],
                                                     IDX["acceleration_reference"],
                                                     ACCELERATION_HARSH_LIMIT,
                                                     "brake_control_usage_harsh"))
    # Compute lateral steering control usage
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_control_usage(data_rdd, IDX["acceleration_cmd"], None, None,
                                                     "steering_control_usage"))
    # Compute lateral steering control usage under harsh curvature
    # Tuple list [(grading items, grading value, sample size)]
    grading_results.append(compute_rdd_control_usage(data_rdd, IDX["acceleration_cmd"],
                                                     IDX["curvature_reference"],
                                                     CURVATURE_HARSH_LIMIT,
                                                     "steering_control_usage_harsh"))

    glog.info('Control performance grading finished with totally {} dimensions'
              .format(len(grading_results)))
    return grading_results
