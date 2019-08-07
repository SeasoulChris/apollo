#!/usr/bin/env

import pprint
import time
import collections
import math
import operator


# Third-party packages
from absl import flags
import colored_glog as glog

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

        self.auto_mileage = 0.0
        self.manual_mileage = 0.0
        self.disengagements = 0

    def run_test(self):
        """Run test."""

        result = (
            # RDD(record_path)
            self.to_rdd(['/apollo/docs/demo_guide/demo_3.5.record'])
            # RDD(PyBagMessage)
            # .flatMap(lambda record: RecordReader(record).read_messages())
            # PairRDD(vehicle_id, auto_mileage, manual_mileage)
            .map(lambda record: self.calculate(record))
            .collect()
        )

        if not result:
            glog.info("Nothing to be processed, everything is under control!")
            return

        pprint.PrettyPrinter().pprint(result)

        glog.info('processed taks:{}'.format(len(result)))

    def run_prod(self):
        """Run prod."""

        origin_prefix = 'small-records/2018'

        return self.run_test()

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
                # glog.info('chassis driving_mode:{}'.format(chassis.driving_mode))
                # Mode changed
                if last_mode != chassis.driving_mode:
                    if (last_mode == Chassis.COMPLETE_AUTO_DRIVE and
                            chassis.driving_mode == Chassis.EMERGENCY_MODE):
                        self.disengagements += 1
                    last_mode = chassis.driving_mode
                    # glog.info('last_mode:{}'.format(last_mode))
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

        return (vehicle_id, auto_mileage, manual_mileage)


if __name__ == '__main__':
    StatMileageByVehicle().main()
