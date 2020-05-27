#!/usr/bin/env python
"""Index records."""
import json

import modules.map.proto.map_pb2 as map_pb2
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.redis_utils as redis_utils
import fueling.common.proto_utils as proto_utils


class IndexMap(BasePipeline):
    """IndexRecords pipeline."""

    def __init__(self):
        self.map_lane_prefix = 'map.data.sunnyvale_with_two_offices.v1.laneid2coord.'
        self.map_file = ("/mnt/bos/code/baidu/adu-lab/apollo-map/"
                         + "sunnyvale_with_two_offices/sim_map.bin")

    def run(self):
        map_pb = map_pb2.Map()
        proto_utils.get_pb_from_file(self.map_file, map_pb)
        for lane in map_pb.lane:
            logging.info("laneid = " + lane.id.id)

            lane_coords = []
            for curve in lane.central_curve.segment:
                if curve.HasField('line_segment'):
                    for p in curve.line_segment.point:
                        x = p.x
                        y = p.y
                        lane_coords.append((x, y))

            redis_utils.redis_set(self.map_lane_prefix + lane.id.id, json.dumps(lane_coords))


if __name__ == '__main__':
    indexing = IndexMap()
    indexing.main()
