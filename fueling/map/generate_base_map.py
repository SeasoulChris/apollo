"""
Run with:
    ./tools/submit-job-to-k8s.py --main=fueling/map/generate_base_map.py --memory=8 --disk=10
"""
#!/usr/bin/env python

# Standard packages
import glob
import time
import os
import math

# Third-party packages
from absl import flags
from shapely.geometry import LineString, Point
import pyspark_utils.helper as spark_helper

# Apollo packages
from cyber_py.record import RecordReader
from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2
from modules.map.proto import map_road_pb2
from modules.localization.proto import localization_pb2

# Apollo-fuel packages
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient

LANE_WIDTH = 3.3

flags.DEFINE_string('input_data_path', 'test/simplehdmap',
                    'simple hdmap input/output data path.')


class MapGenSingleLine(BasePipeline):

    """map_gen_single_line pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'map_gen_single_line')

    def run_test(self):
        """Run test."""
        src_prefix = '/apollo/data/bag'
        dst_prefix = '/apollo/data'
        # RDD(record_path)
        todo_records = self.to_rdd([src_prefix])
        self.run(todo_records, src_prefix, dst_prefix)

        path = os.path.join(dst_prefix, 'base_map.txt')
        if not os.path.exists(path):
            logging.warning('base_map.txt: {} not genterated'.format(path))
        logging.info('base_map.txt generated: Done, Test')

    def run_prod(self):
        dir_prefix = self.FLAGS.get('input_data_path')
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        logging.info("job_id: %s" % job_id)

        #src_prefix = 'simplehdmap/data'
        src_prefix = os.path.join(dir_prefix, 'data')
        dst_prefix = os.path.join(dir_prefix, 'result')

        bos_client = BosClient()
        # Access partner's storage if provided.
        object_storage = self.partner_object_storage() or bos_client

        origin_prefix = os.path.join(dst_prefix, job_owner, job_id)
        target_dir = object_storage.abs_path(origin_prefix)

        if not os.path.exists(target_dir):
            logging.warning('bos path: {} not exists'.format(target_dir))
            file_utils.makedirs(target_dir)
        else:
            logging.info("target_prefix: {}".format(target_dir))

        source_dir = object_storage.abs_path(src_prefix)

        logging.info("source_prefix: {}".format(source_dir))

        # RDD(record_path)
        todo_records = self.to_rdd([source_dir])
        # todo_records = self.to_rdd(glob.glob(os.path.join(source_dir, '*.record*')))
        self.run(todo_records, source_dir, target_dir)

        path = os.path.join(target_dir, 'base_map.txt')
        if not os.path.exists(path):
            logging.warning('base_map.txt: {} not genterated'.format(path))
        logging.info('base_map.txt generated: Done, PROD')

    def run(self, todo_records, src_prefix, dst_prefix):
        """Run the pipeline with given arguments."""
        # Spark cascade style programming.
        self.dst_prefix = dst_prefix
        record_points = spark_helper.cache_and_log('map_gen_single_line',
                                                   # RDD(source_dir)
                                                   todo_records
                                                   # RDD(points)
                                                   .map(self.process_topic)
                                                   # RDD(map)
                                                   .map(self.map_gen))

    def process_topic(self, source_dir):
        points = []
        bi_points = []
        fbags = sorted(glob.glob(os.path.join(source_dir, '*.record*')))
        logging.info('fbags: {}'.format(fbags))
        for fbag in fbags:
            reader = RecordReader(fbag)
            msgs = [msg for msg in reader.read_messages()]

            logging.info('Success to read localization topic message {}'.format(len(msgs)))

            for msg in msgs:
                if msg.topic == "/apollo/localization/pose":
                    localization = localization_pb2.LocalizationEstimate()
                    localization.ParseFromString(msg.message)
                    x = float(localization.pose.position.x)
                    y = float(localization.pose.position.y)
                    points.append((x, y))

        logging.info('Success to read localization pose points {}'.format(len(points)))
        if len(points) == 0:
            return points
        else:
            bi_points.extend(points)
            points.reverse()
            bi_points.extend(points)
            return bi_points
            # map_gen(points)

    def map_gen(self, points):
        logging.info('Success to read localization pose points {}'.format(len(points)))
        path = LineString(points)
        length = int(path.length)

        extra_roi_extension = 1.0

        base_map_txt = os.path.join(self.dst_prefix, 'base_map.txt')
        logging.info("base_map_txt_path: {}".format(base_map_txt))

        fmap = open(base_map_txt, 'w')
        line_id = 0
        base_map = map_pb2.Map()
        road = base_map.road.add()
        road.id.id = "1"
        section = road.section.add()
        section.id.id = "2"
        lane = None
        for i in range(length - 1):
            if i % 100 == 0:
                line_id += 1
                if lane is not None:
                    lane.successor_id.add().id = str(line_id)
                lane, central, left_boundary, right_boundary = self.create_lane(base_map, line_id)
                section.lane_id.add().id = str(line_id)

                left_edge = section.boundary.outer_polygon.edge.add()
                left_edge.type = map_road_pb2.BoundaryEdge.LEFT_BOUNDARY
                left_edge_segment = left_edge.curve.segment.add()

                right_edge = section.boundary.outer_polygon.edge.add()
                right_edge.type = map_road_pb2.BoundaryEdge.RIGHT_BOUNDARY
                right_edge_segment = right_edge.curve.segment.add()

                if i > 0:
                    lane.predecessor_id.add().id = str(line_id - 1)

                    left_bound_point = left_boundary.line_segment.point.add()
                    right_bound_point = right_boundary.line_segment.point.add()
                    central_point = central.line_segment.point.add()

                    right_edge_point = right_edge_segment.line_segment.point.add()
                    left_edge_point = left_edge_segment.line_segment.point.add()

                    point = path.interpolate(i - 1)
                    point2 = path.interpolate(i - 1 + 0.5)
                    distance = LANE_WIDTH / 2.0

                    lp, rp = self.convert(point, point2, distance)
                    left_bound_point.y = lp[1]
                    left_bound_point.x = lp[0]
                    right_bound_point.y = rp[1]
                    right_bound_point.x = rp[0]

                    lp, rp = self.convert(point, point2, distance + extra_roi_extension)
                    left_edge_point.y = lp[1]
                    left_edge_point.x = lp[0]
                    right_edge_point.y = rp[1]
                    right_edge_point.x = rp[0]

                    central_point.x = point.x
                    central_point.y = point.y

                    left_sample = lane.left_sample.add()
                    left_sample.s = 0
                    left_sample.width = LANE_WIDTH / 2.0

                    right_sample = lane.right_sample.add()
                    right_sample.s = 0
                    right_sample.width = LANE_WIDTH / 2.0

            left_bound_point = left_boundary.line_segment.point.add()
            right_bound_point = right_boundary.line_segment.point.add()
            central_point = central.line_segment.point.add()

            right_edge_point = right_edge_segment.line_segment.point.add()
            left_edge_point = left_edge_segment.line_segment.point.add()

            point = path.interpolate(i)
            point2 = path.interpolate(i + 0.5)
            distance = LANE_WIDTH / 2.0
            left_point, right_point = self.convert(point, point2, distance)

            central_point.x = point.x
            central_point.y = point.y
            left_bound_point.y = left_point[1]
            left_bound_point.x = left_point[0]
            right_bound_point.y = right_point[1]
            right_bound_point.x = right_point[0]

            left_edge_point.y = left_point[1]
            left_edge_point.x = left_point[0]
            right_edge_point.y = right_point[1]
            right_edge_point.x = right_point[0]

            left_sample = lane.left_sample.add()
            left_sample.s = i % 100 + 1
            left_sample.width = LANE_WIDTH / 2.0

            right_sample = lane.right_sample.add()
            right_sample.s = i % 100 + 1
            right_sample.width = LANE_WIDTH / 2.0

        fmap.write(str(base_map))
        fmap.close()

        # Output base_map.bin
        base_map_bin = os.path.join(self.dst_prefix, 'base_map.bin')
        logging.info("base_map_bin_path: {}".format(base_map_bin))
        with open(base_map_bin, "wb") as f:
            f.write(base_map.SerializeToString())
        # return str(base_map)

    def convert(self, point, point2, distance):
        delta_y = point2.y - point.y
        delta_x = point2.x - point.x
        # print math.atan2(delta_y, delta_x)
        left_angle = math.atan2(delta_y, delta_x) + math.pi / 2.0
        right_angle = math.atan2(delta_y, delta_x) - math.pi / 2.0
        # print angle
        left_point = []
        left_point.append(point.x + (math.cos(left_angle) * distance))
        left_point.append(point.y + (math.sin(left_angle) * distance))

        right_point = []
        right_point.append(point.x + (math.cos(right_angle) * distance))
        right_point.append(point.y + (math.sin(right_angle) * distance))
        return left_point, right_point

    def shift(self, point, point2, distance, isleft=True):
        delta_y = point2.y - point.y
        delta_x = point2.x - point.x
        # print math.atan2(delta_y, delta_x)
        angle = 0
        if isleft:
            angle = math.atan2(delta_y, delta_x) + math.pi / 2.0
        else:
            angle = math.atan2(delta_y, delta_x) - math.pi / 2.0
        # print angle
        p1n = []
        p1n.append(point.x + (math.cos(angle) * distance))
        p1n.append(point.y + (math.sin(angle) * distance))

        p2n = []
        p2n.append(point2.x + (math.cos(angle) * distance))
        p2n.append(point2.y + (math.sin(angle) * distance))
        return Point(p1n), Point(p2n)

    def create_lane(self, base_map, line_id):
        lane = base_map.lane.add()
        lane.id.id = str(line_id)
        lane.type = map_lane_pb2.Lane.CITY_DRIVING
        lane.direction = map_lane_pb2.Lane.FORWARD
        lane.length = 100.0
        lane.speed_limit = 20.0
        lane.turn = map_lane_pb2.Lane.NO_TURN
        # lane.predecessor_id.add().id = str(line_id - 1)
        # lane.successor_id.add().id = str(line_id + 1)
        left_boundary = lane.left_boundary.curve.segment.add()
        right_boundary = lane.right_boundary.curve.segment.add()
        central = lane.central_curve.segment.add()
        central.length = 100.0

        left_boudary_type = lane.left_boundary.boundary_type.add()
        left_boudary_type.s = 0
        left_boudary_type.types.append(map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW)
        lane.right_boundary.length = 100.0

        right_boudary_type = lane.right_boundary.boundary_type.add()
        right_boudary_type.s = 0
        right_boudary_type.types.append(map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW)
        lane.left_boundary.length = 100.0

        return lane, central, left_boundary, right_boundary


if __name__ == '__main__':
    MapGenSingleLine().main()
