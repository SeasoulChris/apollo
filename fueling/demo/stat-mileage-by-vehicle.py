#!/usr/bin/env

import pprint
import time
import collections
import math
import os
import glob


# Third-party packages
from absl import flags
import colored_glog as glog
import pyspark_utils.op as spark_op

# Apollo packages
from cyber_py.record import RecordReader

from modules.canbus.proto import chassis_pb2
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto import localization_pb2

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils


class StatMileageByVehicle(BasePipeline):
    """pipeline to stat mileage in 2018 with vehicle"""

    def __init__(self):
        BasePipeline.__init__(self, 'stat-mileage-by-vehicle')

    def run_test(self):
        """Run test."""

        demo_record_dir = '/apollo/docs/demo_guide/'
        control_records_bags_dir = '/apollo/modules/data/fuel/testdata/control/control_profiling'

        result = (
            # RDD(record_dirs)
            self.to_rdd(
                [demo_record_dir,
                 os.path.join(control_records_bags_dir, 'Road_Test'),
                 os.path.join(control_records_bags_dir, 'Sim_Test'), ])
            # PairRDD(record_dir, task), the map of target dirs and source dirs
            .keyBy(lambda source: source)
            # PairRDD(record_dirs, record_file)
            .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')) + glob.glob(os.path.join(task, '*bag*')))
            # PairRDD(record_dir, record_file), filter out unqualified files
            .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or record_utils.is_bag_file(file)))
            # PairRDD(record)
            .mapValues(lambda record: self.calculate(record))
            # RDD(auto_mileage)
            .map(lambda x: x[1])
            # RDD(auto_mileages)
            .sum()
        )

        if not result:
            glog.info("Nothing to be processed, everything is under control!")
            return

        pprint.PrettyPrinter().pprint(result)

        glog.info('calculated auto mileage in test is:{}'.format(result))

    def run_prod(self):
        """Run prod."""

        origin_prefix = 'small-records/2018'

        prod_mileage = (
            # RDD(file), start with origin_prefix
            self.to_rdd(self.bos().list_files(origin_prefix))
            # PairRDD(record_dir, task), the map of target dirs and source dirs
            .keyBy(lambda source: source)
            # PairRDD(record_dirs, record_file)
            .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')) + glob.glob(os.path.join(task, '*bag*')))
            # PairRDD(record_dir, record_file), filter out unqualified files
            .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or record_utils.is_bag_file(file)))
            # PairRDD(record)
            .mapValues(lambda record: self.calculate(record))
            # RDD(auto_mileage)
            .map(lambda x: x[1])
            # RDD(auto_mileage)
            .sum()
        )

        glog.info(
            'calculated auto mileage in production is :{}'.format(prod_mileage))

    def calculate(self, record):
        """Calculate mileage"""

        last_pos = None
        last_mode = 'Unknown'
        mileage = collections.defaultdict(lambda: 0.0)
        chassis = chassis_pb2.Chassis()
        localization = localization_pb2.LocalizationEstimate()
        vehicle_id = None

        auto_mileage = 0.0
        manual_mileage = 0.0
        disengagements = 0

        reader = record_utils.read_record(
            [record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL])
        for msg in reader(record):
            if msg.topic == record_utils.HMI_STATUS_CHANNEL:
                hmi_status = record_utils.message_to_proto(msg)
                if hmi_status is None:
                    continue
                vehicle_id = hmi_status.current_vehicle
            elif msg.topic == record_utils.CHASSIS_CHANNEL:
                chassis = record_utils.message_to_proto(msg)
                # Mode changed
                if last_mode != chassis.driving_mode:
                    if (last_mode == Chassis.COMPLETE_AUTO_DRIVE and
                            chassis.driving_mode == Chassis.EMERGENCY_MODE):
                        disengagements += 1
                    last_mode = chassis.driving_mode
                    # Reset start position.
                    last_pos = None
            elif msg.topic == record_utils.LOCALIZATION_CHANNEL:
                localization = record_utils.message_to_proto(msg)
                cur_pos = localization.pose.position
                if last_pos:
                    # Accumulate mileage, from xyz-distance(m) to miles.
                    mileage[last_mode] += 0.000621371 * math.sqrt(
                        (cur_pos.x - last_pos.x) ** 2 +
                        (cur_pos.y - last_pos.y) ** 2 +
                        (cur_pos.z - last_pos.z) ** 2)
                last_pos = cur_pos

        auto_mileage += mileage[Chassis.COMPLETE_AUTO_DRIVE]
        manual_mileage += (mileage[Chassis.COMPLETE_MANUAL] +
                           mileage[Chassis.EMERGENCY_MODE])

        # vehicle_id is None in most cases, so not returned
        return auto_mileage


if __name__ == '__main__':
    StatMileageByVehicle().main()
