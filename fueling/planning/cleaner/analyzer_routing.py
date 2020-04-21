#!/usr/bin/env python
"""Routing Analyzer."""

import math

from shapely.geometry import LineString
from shapely.geometry import Point

from modules.routing.proto.routing_pb2 import RoutingResponse
import fueling.common.logging as logging

from fueling.planning.map_reader.map_reader import MapReader


class RoutingAnalyzer:
    def __init__(self):
        self.routing_response = None
        self.routing_response_msg = None
        self.routing_lines = []

    def update(self, routing_response_msg):
        self.routing_response_msg = routing_response_msg
        self.routing_response = RoutingResponse()
        self.routing_response.ParseFromString(routing_response_msg.message)
        self._get_routing_lines()

    def get_routing_response_msg(self):
        return self.routing_response_msg

    def get_routing_response(self):
        return self.routing_response

    def is_adv_on_routing(self, localization_estimate):
        if len(self.routing_lines) == 0:
            return False

        x = localization_estimate.pose.position.x
        y = localization_estimate.pose.position.y
        p = Point(x, y)

        mini_dist = 999999999
        for routing_line in self.routing_lines:
            dist = LineString(routing_line).distance(p)
            if dist < mini_dist:
                mini_dist = dist

        if mini_dist > 1.5:
            return False

        return True

    def _get_routing_lines(self):
        self.routing_lines = []

        map_reader = MapReader()
        for road in self.routing_response.road:
            for passage in road.passage:
                passageline = []
                for segment in passage.segment:
                    coords = map_reader.lane_id_to_coords(segment.id)
                    if coords is None:
                        logging.info("didn't found lane id: " + segment.id)
                    else:
                        passageline.extend(coords)
                self._merge_passageline_into_lines(passageline)

    def _merge_passageline_into_lines(self, passageline):
        if len(passageline) == 0:
            return

        found = False
        for line in self.routing_lines:
            p1 = line[-1]
            p2 = passageline[0]
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist <= 0.3:
                line.extend(passageline)
                found = True
                break
        if not found:
            self.routing_lines.append(passageline)


if __name__ == "__main__":
    import sys
    from cyber_py3.record import RecordReader

    fn = sys.argv[1]
    reader = RecordReader(fn)
    msgs = [msg for msg in reader.read_messages()]
    analyzer = RoutingAnalyzer()
    analyzer.analyze(msgs)
