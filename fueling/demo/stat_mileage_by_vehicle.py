#!/usr/bin/env python

import collections
import glob
import os

# Third-party packages
import colored_glog as glog

# Apollo packages
from modules.canbus.proto import chassis_pb2
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto import localization_pb2

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.data.record_parser as record_parser


class StatMileageByVehicle(BasePipeline):
    """pipeline to stat mileage with vehicle"""

    def __init__(self):
        BasePipeline.__init__(self, 'stat-mileage-by-vehicle')

    def run_test(self):
        """Run test."""

        demo_record_dir = '/apollo/docs/demo_guide/'
        control_records_bags_dir = '/apollo/modules/data/fuel/testdata/control/control_profiling'

        test_dirs = (
            # RDD(record_dir)
            self.to_rdd([
                demo_record_dir,
                os.path.join(control_records_bags_dir, 'Road_Test'),
                os.path.join(control_records_bags_dir, 'Sim_Test'),
            ])
            # RDD(record_file)
            .flatMap(lambda dir: glob.glob(os.path.join(dir, '*record*')) +
                     glob.glob(os.path.join(dir, '*bag*'))))
        result = self.run(test_dirs)

        glog.info('Calculated auto mileage in test mode is:{}'.format(result))

    def run_prod(self):
        """Run prod."""

        origin_prefix = 'small-records/2018/2018-04-03'
        # RDD(record_file)
        todo_dirs = (
            self.to_rdd(self.bos().list_files(origin_prefix))
            # RDD(record_file)
            .map(lambda dir: glob.glob(os.path.join(dir, '*record*')) + glob.
                 glob(os.path.join(dir, '*bag*'))))

        prod_mileage = self.run(todo_dirs)

        glog.info('Calculated auto mileage in production mode is :{}'.format(
            prod_mileage))

    def run(self, dirs):
        result = (
            dirs
            # RDD(record_file), filter out unqualified files
            .filter(lambda file: record_utils.is_record_file(file) or
                    record_utils.is_bag_file(file))
            # RDD(auto_mileage)
            .map(self.calculate)
            # float sum auto_mileage
            .sum()
        )

        if not result:
            glog.info("Nothing to be processed, everything is under control!")
            return None

        return result

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
                    # Accumulate mileage, from meters to miles.
                    mileage[last_mode] += 0.000621371 * \
                        record_parser.pose_distance_m(cur_pos, last_pos)
                last_pos = cur_pos

        auto_mileage += mileage[Chassis.COMPLETE_AUTO_DRIVE]
        manual_mileage += (mileage[Chassis.COMPLETE_MANUAL] +
                           mileage[Chassis.EMERGENCY_MODE])

        # vehicle_id is None in most cases, so not returned
        return auto_mileage


if __name__ == '__main__':
    StatMileageByVehicle().main()
