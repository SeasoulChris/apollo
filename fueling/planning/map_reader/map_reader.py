#!/usr/bin/env python

import json
import fueling.common.redis_utils as redis_utils


class MapReader:
    def __init__(self, map_name="sunnyvale_with_two_offices"):
        self.maps = ["sunnyvale_with_two_offices"]
        if map_name in map_name:
            self.map_lane_prefix = 'map.data.' + map_name + '.v1.laneid2coord.'

    def lane_id_to_coords(self, lane_id):
        coords = redis_utils.redis_get(self.map_lane_prefix + lane_id)
        if coords is None:
            return None
        return json.loads(coords)
