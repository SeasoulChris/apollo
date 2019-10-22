#!/usr/bin/env python
"""
A simple demo to parse records and stat auto mileage.

Run with:
    ./tools/submit-job-to-k8s.py --entrypoint=fueling/demo/stat_auto_mileage.py
"""

import glob

# Apollo packages
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto.localization_pb2 import LocalizationEstimate

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.data.record_parser as record_parser


class StatAutoMileage(BasePipeline):
    """pipeline to stat auto mileage"""

    def __init__(self):
        BasePipeline.__init__(self, 'stat-auto-mileage')

    def run_test(self):
        """Run test."""

        # RDD(record)
        records = self.to_rdd(glob.glob(
            '/apollo/modules/data/fuel/testdata/control/control_profiling/Road_Test/*record*'))
        self.run(records)

    def run_prod(self):
        """Run prod."""
        # RDD(record)
        records = (
            self.to_rdd(self.bos().list_files('small-records/2018/2018-04-03/2018-04-03-09-33-50'))
            .filter(record_utils.is_record_file))
        self.run(records)

    def run(self, records):
        mileage = records.map(self.calculate).sum()
        logging.info('Calculated auto mileage is: {}'.format(mileage))

    def calculate(self, record):
        """Calculate mileage"""
        last_pos = None
        last_mode = 'Unknown'
        chassis = Chassis()
        localization = LocalizationEstimate()

        auto_mileage = 0.0
        reader = record_utils.read_record(
            [record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL])
        for msg in reader(record):
            if msg.topic == record_utils.CHASSIS_CHANNEL:
                chassis = record_utils.message_to_proto(msg)
                # Mode changed
                if last_mode != chassis.driving_mode:
                    last_mode = chassis.driving_mode
                    # Reset start position.
                    last_pos = None
            elif msg.topic == record_utils.LOCALIZATION_CHANNEL:
                localization = record_utils.message_to_proto(msg)
                cur_pos = localization.pose.position
                if last_pos and last_mode == Chassis.COMPLETE_AUTO_DRIVE:
                    # Accumulate mileage, from meters to miles.
                    auto_mileage += 0.000621371 * record_parser.pose_distance_m(cur_pos, last_pos)
                last_pos = cur_pos
        return auto_mileage


if __name__ == '__main__':
    StatAutoMileage().main()
