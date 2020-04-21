#!/usr/bin/env python

import math
from shapely.geometry import LineString
from shapely.geometry import Point

from modules.localization.proto import localization_pb2
from modules.routing.proto.routing_pb2 import RoutingResponse
import fueling.common.logging as logging

from fueling.planning.map_reader.map_reader import MapReader


class AgentAnalyzer:
    def __init__(self):
        self.map_reader = MapReader()
        self.line_strings = []

    def process(self, msgs):
        filtered_msgs = []

        for msg in msgs:
            if msg.topic == "/apollo/routing_response":
                routing_response = RoutingResponse()
                routing_response.ParseFromString(msg.message)
                lines = self.routing_response_to_lines(routing_response)
                logging.info("~~~~ found lines = " + str(len(lines)))
                for line in lines:
                    self.line_strings.append(LineString(line))

            if msg.topic == '/apollo/localization/pose':
                localization_estimate = localization_pb2.LocalizationEstimate()
                localization_estimate.ParseFromString(msg.message)
                x = localization_estimate.pose.position.x
                y = localization_estimate.pose.position.y
                p = Point(x, y)
                if len(self.line_strings) == 0:
                    continue
                mini_dist = 999999999
                for linestring in self.line_strings:
                    dist = linestring.distance(p)
                    if dist < mini_dist:
                        mini_dist = dist
                if mini_dist > 5:
                    continue
            filtered_msgs.append(msg)

        return filtered_msgs

    def routing_response_to_lines(self, routing_resp):
        map_reader = MapReader()
        lines = []
        for road in routing_resp.road:
            for passage in road.passage:
                passageline = []
                for segment in passage.segment:
                    coords = map_reader.lane_id_to_coords(segment.id)
                    if coords is None:
                        logging.info("didn't found lane id: " + segment.id)
                    else:
                        passageline.extend(coords)
                self.merge_passageline_into_lines(lines, passageline)
        return lines

    def merge_passageline_into_lines(self, lines, passageline):
        found = False
        for line in lines:
            p1 = line[-1]
            p2 = passageline[0]
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist <= 0.3:
                line.extend(passageline)
                found = True
                break
        if not found:
            lines.append(passageline)
