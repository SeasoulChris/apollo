#!/usr/bin/env python
"""
This is a module to extraction logsim scenarios from records
based on disengage info
"""
import glob
import os

from cyber_py.record import RecordReader
from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
from modules.routing.proto.routing_pb2 import RoutingResponse


class ScenarioExtractionPipeline(BasePipeline):
    """Extract logsim scenarios from records and save the bag/json pair"""

    def __init__(self):
        """Initialize"""
        BasePipeline.__init__(self, 'scenario_extraction')

    def run_test(self):
        """Local mini test."""
        local_prefix = '/apollo/modules/data/fuel/testdata/modules/'
        input_dir = local_prefix + 'scenario_extraction_raw/'
        output_dir = local_prefix + 'scenario_extraction_logsim/'

        map_dir = 'modules/map/data/san_mateo'
        records = glob.glob(input_dir + "/*.record.?????")
        for msg in RecordReader(records[0]).read_messages():
            if msg.topic == '/apollo/routing_response_history':
                proto = RoutingResponse()
                proto.ParseFromString(msg.message)
                if proto.map_version.startswith('sunnyvale'):
                    map_dir = 'modules/map/data/sunnyvale_with_two_offices'
                break

        self.run(input_dir, output_dir, map_dir)

    def run_prod(self, input_dir):
        """Work on actual road test data. Expect a single input directory"""
        origin_prefix = 'small-records/2019/'
        target_prefix = 'logsim_scenarios/2019/'
        output_dir = input_dir.replace(origin_prefix, target_prefix)

        # figure out the map used in this road test
        map_dir = 'modules/map/data/san_mateo'
        files = s3_utils.list_objects('apollo-platform', input_dir)
        records = [f.Key for f in files if record_utils.is_record_file(f.Key)]

        for msg in RecordReader(records[0]).read_messages():
            if msg.topic == '/apollo/routing_response_history':
                proto = RoutingResponse()
                proto.ParseFromString(msg.message)
                if proto.map_version.startswith('sunnyvale'):
                    map_dir = 'modules/map/data/sunnyvale_with_two_offices'
                break

        self.run(input_dir, output_dir, map_dir)

    def run(self, input_dir, output_dir, map_dir):
        """Invoking logsim_generator binary"""
        print ("Start to extract logsim scenarios for %s and map %s" %
               (input_dir, map_dir))
        os.system('bash %s/logsim_generator.sh %s %s %s' %
                  (os.path.dirname(os.path.realpath(__file__)),
                   input_dir, output_dir, map_dir))


if __name__ == '__main__':
    ScenarioExtractionPipeline().run_test()
