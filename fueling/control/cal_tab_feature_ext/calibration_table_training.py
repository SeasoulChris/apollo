from collections import Counter
import glob
import h5py
import operator
import os


from neural_network_tf import NeuralNetworkTF
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import common.proto_utils as proto_utils
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.calibration_table_train_utils as calibration_table_train_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import modules.control.proto.calibration_table_pb2 as calibration_table_pb2
import modules.control.proto.control_conf_pb2 as ControlConf
import modules.data.fuel.fueling.control.proto.calibration_table_pb2 as calibrationTable


WANTED_VEHICLE = 'Transit'

FILENAME_CALIBRATION_TABLE_CONF = os.path.join(os.path.dirname(__file__),
                                               '../conf/calibration_table_conf.pb.txt'))
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(FILENAME_CALIBRATION_TABLE_CONF,
                                                           calibrationTable.calibrationTable())

FILENAME_CONTROL_CONF = "/mnt/bos/code/apollo-internal/modules_data/calibration/data/transit/<TODO>"
CONTROL_CONF = proto_utils.get_pb_from_text_file(FILENAME_CONTROL_CONF, ControlConf.ControlConf())


brake_train_layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
                     CALIBRATION_TABLE_CONF.brake_train_layer2,
                     CALIBRATION_TABLE_CONF.brake_train_layer3]
throttle_train_layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
                        CALIBRATION_TABLE_CONF.throttle_train_layer2,
                        CALIBRATION_TABLE_CONF.throttle_train_layer3]
train_alpha = CALIBRATION_TABLE_CONF.train_alpha

brake_axis_cmd_min = -1*CONTROL_CONF.lon_controller_conf.brake_deadzone
brake_axis_cmd_max = -1*CALIBRATION_TABLE_CONF.brake_max

speed_min = CALIBRATION_TABLE_CONF.train_speed_min
speed_max = CALIBRATION_TABLE_CONF.train_speed_min
speed_segment_num = CALIBRATION_TABLE_CONF.train_speed_segment

throttle_axis_cmd_min = CONTROL_CONF.lon_controller_conf.throttle_deadzone
throttle_axis_cmd_max = CALIBRATION_TABLE_CONF.throttle_max
cmd_segment_num = CALIBRATION_TABLE_CONF.train_cmd_segment


class CalibrationTableTraining(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_training')

    def run_test(self):
        """Run test."""
        records = [
            'modules/data/fuel/testdata/control/']

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = 'modules/data/fuel/testdata/control/generated'
        root_dir = '/apollo'
        dir_to_records = self.get_spark_context().parallelize(
            records).keyBy(os.path.dirname)

        self.run(dir_to_records, origin_prefix, target_prefix, root_dir)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = 'modules/control/feature_extraction_hf5/2019/'
        root_dir = s3_utils.S3_MOUNT_PATH

        files = s3_utils.list_files(bucket, origin_prefix).cache()
        complete_dirs = files.filter(
            lambda path: path.endswith('/COMPLETE')).map(os.path.dirname)
        dir_to_records = files.filter(
            record_utils.is_record_file).keyBy(os.path.dirname)
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """

        # -> (dir, record), in absolute path
        dir_to_records = dir_to_records_rdd.map(lambda x: (os.path.join(root_dir, x[0]),
                                                           os.path.join(root_dir, x[1]))).cache()
        print(dir_to_records.first())

        throttle_train_file_rdd = (dir_to_records
                                   # training data (hdf5 file) vehicle
                                   .map(lambda elem:
                                        calibration_table_train_utils.choose_data_file(elem, WANTED_VEHICLE, 'throttle', 'train'))
                                   # generate training data segment
                                   .mapValues(calibration_table_train_utils.generate_segments)
                                   #   generate training data: x_train_data, y_train_data
                                   .mapValues(calibration_table_train_utils.generate_data)).cache()

        throttle_test_file_rdd = (dir_to_records
                                  # training data (hdf5 file) vehicle
                                  .map(lambda elem:
                                       calibration_table_train_utils.choose_data_file(elem, WANTED_VEHICLE, 'throttle', 'test'))
                                  # generate training data segment
                                  .mapValues(calibration_table_train_utils.generate_segments)
                                  #   generate training data: x_train_data, y_train_data
                                  .mapValues(calibration_table_train_utils.generate_data)).cache()

        throttle_table_filename = WANTED_VEHICLE + '_throttle_calibration_table.pb.txt'
        throttle_model_rdd = (throttle_train_file_rdd
                              .join(throttle_test_file_rdd)
                              .mapValues(lambda elem:
                                         calibration_table_train_utils.train_model(elem, throttle_train_layer, train_alpha))
                              .map(lambda elem:
                                   calibration_table_train_utils.write_table(elem,
                                                                             speed_min, speed_max, speed_segment_num,
                                                                             throttle_axis_cmd_min, throttle_axis_cmd_max, cmd_segment_num,
                                                                             throttle_table_filename)))

        throttle_model_rdd.collect()
        print(throttle_model_rdd.first())

        brake_train_file_rdd = (dir_to_records
                                # training data (hdf5 file) vehicle
                                .map(lambda elem:
                                     calibration_table_train_utils.choose_data_file(elem, WANTED_VEHICLE, 'brake', 'train'))
                                # generate training data segment
                                .mapValues(calibration_table_train_utils.generate_segments)
                                #   generate training data: x_train_data, y_train_data
                                .mapValues(calibration_table_train_utils.generate_data)).cache()

        brake_test_file_rdd = (dir_to_records
                               # training data (hdf5 file) vehicle
                               .map(lambda elem:
                                    calibration_table_train_utils.choose_data_file(elem, WANTED_VEHICLE, 'brake', 'test'))
                               # generate training data segment
                               .mapValues(calibration_table_train_utils.generate_segments)
                               #   generate training data: x_train_data, y_train_data
                               .mapValues(calibration_table_train_utils.generate_data)).cache()

        brake_table_filename = WANTED_VEHICLE+'_brake_calibration_table.pb.txt'
        brake_model_rdd = (brake_train_file_rdd
                           .join(brake_test_file_rdd)
                           .mapValues(lambda elem:
                                      calibration_table_train_utils.train_model(elem, brake_train_layer, train_alpha))
                           .map(lambda elem:
                                calibration_table_train_utils.write_table(elem,
                                                                          speed_min, speed_max, speed_segment_num,
                                                                          brake_axis_cmd_min, brake_axis_cmd_max, cmd_segment_num,
                                                                          brake_table_filename)))

        brake_model_rdd.collect()
        print(brake_model_rdd.first())


if __name__ == '__main__':
    CalibrationTableTraining().run_test()
