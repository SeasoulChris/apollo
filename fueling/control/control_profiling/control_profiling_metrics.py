#!/usr/bin/env python
""" Extracting features and grading the control performance based on the designed metrics """
# pylint: disable = fixme
# pylint: disable = no-member

from collections import Counter
import operator
import os

import pyspark_utils.op as spark_op

import common.proto_utils as proto_utils
import modules.control.proto.control_conf_pb2 as ControlConf
import modules.data.fuel.fueling.control.proto.control_profiling_pb2 \
    as control_profiling_conf

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.control_profiling.feature_extraction.control_feature_extraction_utils \
    as feature_utils
import fueling.control.control_profiling.grading_evaluation.control_performance_grading_utils \
    as grading_utils


# Extract the vehicle-related setting from the control_profiling_conf.pb.txt
FILENAME_CONTROL_PROFILING = \
    "/apollo/modules/data/fuel/fueling/control/conf/control_profiling_conf.pb.txt"
CONTROL_PROFILING = control_profiling_conf.ControlProfiling()
proto_utils.get_pb_from_text_file(FILENAME_CONTROL_PROFILING, CONTROL_PROFILING)

# Extract the control-related setting from the control_conf.pb.txt
FILENAME_CONTROL_CONF = "/apollo/modules/control/conf/control_conf.pb.txt"
CONTROL_CONF = ControlConf.ControlConf()
proto_utils.get_pb_from_text_file(FILENAME_CONTROL_CONF, CONTROL_CONF)

# Define the target vehicle type and controller type based on the selected records
VEHICLE_TYPE = CONTROL_PROFILING.vehicle_type
CONTROLLER_TYPE = CONTROL_PROFILING.controller_type
ACTIVE_CONTROLLERS = CONTROL_CONF.active_controllers

# Define other important variabels
MIN_MSG_PER_SEGMENT = 100


class ControlProfilingMetrics(BasePipeline):
    """ Control Profiling: Feature Extraction and Performance Grading """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'control_profiling_metrics')

    def run_test(self):
        """Run test."""
        origin_prefix = 'modules/data/fuel/testdata/control/control_profiling'
        target_prefix = 'modules/data/fuel/testdata/control/control_profiling/generated'
        root_dir = '/apollo'
        records = [
            origin_prefix + "/Transit_Auto/20190225162427.record.00001",
            origin_prefix + "/Transit_Auto/20190225162427.record.00002",
            origin_prefix + "/Transit_Auto2/20190225162427.record.00003",
        ]
        # PairRDD(dir_path, dir_record)
        dir_to_records = self.get_spark_context().parallelize(
            records).keyBy(os.path.dirname)
        self.run(dir_to_records, origin_prefix, target_prefix, root_dir)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = os.path.join('modules/control/control_profiling_hf5/',
                                     VEHICLE_TYPE, 'SampleSet')
        root_dir = s3_utils.S3_MOUNT_PATH
        files = s3_utils.list_files(bucket, origin_prefix).cache()
        complete_dirs = files.filter(
            lambda path: path.endswith('/COMPLETE')).map(os.path.dirname)
        dir_to_records = files.filter(
            record_utils.is_record_file).keyBy(os.path.dirname)
        root_dir = s3_utils.S3_MOUNT_PATH
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD and grading the control variables"""

        # String
        controller_verify = feature_utils.compare_controller_type(CONTROLLER_TYPE,
                                                                  ACTIVE_CONTROLLERS,
                                                                  CONTROL_CONF)


        dir_to_records = (
            # PairRDD(dir, record)
            dir_to_records_rdd
            # PairRDD(dir, record_path) under the root directory
            .map(lambda x: (os.path.join(root_dir, x[0]), os.path.join(root_dir, x[1])))
            .cache())

        selected_vehicles = (
            # PairRDD(dir, vehicle_type)
            feature_utils.get_vehicle_of_dirs(dir_to_records)
            # PairRDD(dir, selected_vehicle_type)
            .filter(spark_op.filter_value(lambda vehicle: vehicle == VEHICLE_TYPE))
            # RDD(dir) which include records of the selected vehicle
            .keys())

        selected_controllers = (
            # PairRDD(dir, controller_type)
            feature_utils.get_controller_of_dirs(dir_to_records)
            # PairRDD(dir, selected_controller_type)
            .filter(spark_op.filter_value(lambda vehicle: vehicle == CONTROLLER_TYPE))
            # RDD(dir) which include records of the selected controller
            .keys())

        # RDD(dir) which include records of the selected vehicle and controller
        selected_dir = selected_vehicles.intersection(selected_controllers)

        glog.info('{}'.format(selected_dir.collect()))

        # Selcted Channel_to_type
        channels = {record_utils.CONTROL_CHANNEL}

        dir_to_msgs = (
            # PairRDD(dir, record_path) under the root directory
            # which include records of the selected vehicle and controller
            spark_op.filter_keys(dir_to_records, selected_dir)
            # PairRDD(dir, msg)
            .flatMapValues(record_utils.read_record(channels))
            # PairRDD(dir_segment, msg)
            .map(feature_utils.gen_pre_segment)
            .cache())

        valid_segments = (
            dir_to_msgs
            # PairRDD(dir_segment, topic_counter)
            .mapValues(lambda msg: Counter([msg.topic]))
            # PairRDD(dir_segment, total count)
            .reduceByKey(operator.add)
            # PairRDD(dir_segment, total count)
            .filter(spark_op.filter_value(
                lambda counter:
                counter.get(record_utils.CONTROL_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT))
            # RDD(dir_segment)
            .keys())

        data_segment_rdd = (
            # PairRDD((dir, timestamp_per_min), msg)
            spark_op.filter_keys(dir_to_msgs, valid_segments)
            # PairRDD ((dir, timestamp_per_min), msgs)
            .groupByKey()
            # PairRDD((dir, timestamp_per_min), proto_dict)
            .mapValues(record_utils.messages_to_proto_dict())
            # PairRDD((dir, timestamp_per_min), list of the whole selected channel)
            .flatMapValues(lambda proto_dict: proto_dict[record_utils.CONTROL_CHANNEL])
            # PairRDD((dir, timestamp_sec), list of the selected variables)
            .map(feature_utils.extract_data(CONTROLLER_TYPE))
            .cache())

        data_segment_h5 = (
            data_segment_rdd
            # PairRDD((dir, controller_type), list of the selected variables)
            .map(lambda x: ((x[0][0], CONTROLLER_TYPE), x[1]))
            # PairRDD((dir, controller_type), combined list of the selected variables)
            .combineByKey(feature_utils.to_list, feature_utils.append,
                          feature_utils.extend)
            # PairRDD((dir, controller_type),  segments)
            .mapValues(feature_utils.gen_segment_by_key)
            # RDD(dir, controller_type), write all segment into a hdf5 file
            .map(lambda elem:
                 feature_utils.write_h5_with_key(elem, origin_prefix, target_prefix, VEHICLE_TYPE)))

        grading_output = (
            # Tuple list [(grading Items, grading value, sample size)]
            grading_utils.performance_grading(data_segment_rdd))

        glog.info('The data segments transferred into h5 files: {}'
                  .format(data_segment_h5.collect()))

        for i in range(1, len(grading_output)):
            glog.info('grading_output: {} {:f} {}'
                      .format(grading_output[i][0], grading_output[i][1], grading_output[i][2]))

        # Define the grading results .txt output path
        grading_result_path = (
            os.path.join(os.path.join(root_dir, target_prefix),
                         VEHICLE_TYPE + '_' + CONTROLLER_TYPE + '_control_performance_grading.txt'))

        # TODO(Yu): build a template to encapsulate a clean and compact data writing process
        # Export the control profiling outputs as a .txt file
        with open(grading_result_path, 'w') as txt:
            txt.write('\nControl Profiling Grading Summary:\n\n')
            txt.write('Grading on vehicle: \t {} \n' .format(VEHICLE_TYPE))
            txt.write('Grading on controller: \t {} \n' .format(CONTROLLER_TYPE))
            txt.write('Grading on record path: \t {} \n'
                      .format(dir_to_records.groupByKey().map(lambda x: x[0]).collect()))
            txt.write('Grading_output: \t {0:<32s} {1:<16s} {2:<16s} \n'
                      .format(grading_output[0][0], grading_output[0][1], grading_output[0][2]))
            for i in range(1, len(grading_output)):
                txt.write('Grading_output: \t {0:<32s} {1:<16,.3%} {2:<16n} \n'
                          .format(grading_output[i][0], grading_output[i][1], grading_output[i][2]))
            txt.write('\n\n\nMetrics in control_profiling_conf.pb.txt File:\n\n')
            txt.write('{}'.format(CONTROL_PROFILING))

if __name__ == '__main__':
    ControlProfilingMetrics().run_test()
