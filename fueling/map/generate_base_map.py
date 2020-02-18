"""
Run with:
    ./tools/submit-job-to-k8s.py --main=fueling/map/generate_base_map.py --memory=8 --disk=10
"""
#!/usr/bin/env python

# Standard packages
import datetime
import glob
import math
import os

# Third-party packages
from shapely.geometry import LineString, Point
import pyspark_utils.helper as spark_helper

# Apollo packages
from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2
from modules.map.proto import map_road_pb2

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
from modules.data.fuel.apps.web_portal.saas_job_arg_pb2 import SaasJobArg
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.redis_utils as redis_utils

LANE_WIDTH = 3.3


class MapGenSingleLine(BasePipeline):
    """map_gen_single_line pipeline."""

    def run_test(self):
        """Run test."""
        dir_prefix = '/apollo/data/bag'
        src_prefix = os.path.join(dir_prefix, 'data')
        dst_prefix = os.path.join(dir_prefix, 'result')
        # RDD(record_path)
        todo_records = self.to_rdd([src_prefix])
        self.run(todo_records, src_prefix, dst_prefix)

        path = os.path.join(dst_prefix, 'base_map.txt')
        if not os.path.exists(path):
            logging.warning('base_map.txt: {} not genterated'.format(path))
        logging.info('base_map.txt generated: Done, Test')

    def run_prod(self):
        src_prefix = self.FLAGS.get('input_data_path', 'test/virtual_lane/data')
        dst_prefix = self.FLAGS.get('output_data_path', 'test/virtual_lane/result')

        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        logging.info("job_id: %s" % job_id)

        if src_prefix == dst_prefix:
            logging.error('The input data path must be different from the output data path!')
            return

        # Access partner's storage if provided.
        object_storage = self.partner_storage() or self.our_storage()

        origin_prefix = os.path.join(dst_prefix, job_owner, job_id)
        target_dir = object_storage.abs_path(origin_prefix)

        if not os.path.exists(target_dir):
            logging.warning('bos path: {} not exists'.format(target_dir))
            file_utils.makedirs(target_dir)
        else:
            logging.info("target_prefix: {}".format(target_dir))

        source_dir = object_storage.abs_path(src_prefix)

        logging.info("source_prefix: {}".format(source_dir))

        job_type, job_size = SaasJobArg.VIRTUAL_LANE_GENERATION, file_utils.getDirSize(source_dir)
        redis_key = F'External_Partner_Job.{job_owner}.{job_type}.{job_id}'
        redis_value = {'begin_time': datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                       'job_size': job_size,
                       'job_status': 'running'}
        redis_utils.redis_extend_dict(redis_key, redis_value)


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
        spark_helper.cache_and_log('map_gen_single_line',
                                   # RDD(source_dir)
                                   todo_records
                                   # RDD(points)
                                   .map(self.process_topic)
                                   # RDD(map)
                                   .map(self.map_gen))

    def process_topic(self, source_dir):
        points = []
        # bi_points = []
        fbags = sorted(glob.glob(os.path.join(source_dir, '*.record*')))
        logging.info('fbags: {}'.format(fbags))
        reader = record_utils.read_record([record_utils.LOCALIZATION_CHANNEL])
        for fbag in fbags:
            for msg in reader(fbag):
                pos = record_utils.message_to_proto(msg).pose.position
                points.append((pos.x, pos.y))

        logging.info('Success to read localization pose points {}'.format(len(points)))
        return points
        # map_gen(points)

    def map_gen(self, points):
        logging.info('Success to read localization pose points {}'.format(len(points)))
        path = LineString(points)
        points.reverse()
        path_dup = LineString(points)
        length = int(path.length)

        extra_roi_extension = 1.0

        base_map_txt = os.path.join(self.dst_prefix, 'base_map.txt')
        logging.info("base_map_txt_path: {}".format(base_map_txt))

        fmap = open(base_map_txt, 'w')
        lane = None
        line_id = 0
        road_id = 0
        base_map = map_pb2.Map()
        self.lane_length = length

        for j in range(2):
            if j == 1:
                path = path_dup
            road = base_map.road.add()
            road_id += 1
            road.id.id = str(road_id)
            section = road.section.add()
            section.id.id = "2"
            for i in range(length - 1):
                if i % self.lane_length == 0:
                    line_id += 1
                    # if lane is not None:
                    #     lane.successor_id.add().id = str(line_id)
                    lane, central, left_boundary, right_boundary = self.create_lane(base_map, line_id)
                    section.lane_id.add().id = str(line_id)

                    left_edge = section.boundary.outer_polygon.edge.add()
                    left_edge.type = map_road_pb2.BoundaryEdge.LEFT_BOUNDARY
                    left_edge_segment = left_edge.curve.segment.add()

                    right_edge = section.boundary.outer_polygon.edge.add()
                    right_edge.type = map_road_pb2.BoundaryEdge.RIGHT_BOUNDARY
                    right_edge_segment = right_edge.curve.segment.add()
                    if j == 0:
                        lane.self_reverse_lane_id.add().id = "2"
                    else:
                        lane.self_reverse_lane_id.add().id = "1"

                    if i > 0:
                        # lane.predecessor_id.add().id = str(line_id - 1)                         

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

                if i == 0:
                    central_start_pos = central.start_position
                    left_bound_start_pos = left_boundary.start_position
                    right_bound_start_pos = right_boundary.start_position
                    
                    central_start_pos.x = central_point.x
                    central_start_pos.y = central_point.y                    
                    left_bound_start_pos.x = left_bound_point.x
                    left_bound_start_pos.y = left_bound_point.y
                    right_bound_start_pos.x = right_bound_point.x
                    right_bound_start_pos.y = right_bound_point.y

                left_edge_point.y = left_point[1]
                left_edge_point.x = left_point[0]
                right_edge_point.y = right_point[1]
                right_edge_point.x = right_point[0]               

                left_sample = lane.left_sample.add()
                left_sample.s = i % self.lane_length + 1
                left_sample.width = LANE_WIDTH / 2.0

                right_sample = lane.right_sample.add()
                right_sample.s = i % self.lane_length + 1
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
        # print(math.atan2(delta_y, delta_x))
        left_angle = math.atan2(delta_y, delta_x) + math.pi / 2.0
        right_angle = math.atan2(delta_y, delta_x) - math.pi / 2.0
        # print(angle)
        left_point = [
            point.x + (math.cos(left_angle) * distance),
            point.y + (math.sin(left_angle) * distance),
        ]
        right_point = [
            point.x + (math.cos(right_angle) * distance),
            point.y + (math.sin(right_angle) * distance),
        ]
        return left_point, right_point

    def shift(self, point, point2, distance, isleft=True):
        delta_y = point2.y - point.y
        delta_x = point2.x - point.x
        # print(math.atan2(delta_y, delta_x))
        angle = 0
        if isleft:
            angle = math.atan2(delta_y, delta_x) + math.pi / 2.0
        else:
            angle = math.atan2(delta_y, delta_x) - math.pi / 2.0
        # print(angle)
        p1n = [
            point.x + (math.cos(angle) * distance),
            point.y + (math.sin(angle) * distance),
        ]
        p2n = [
            point2.x + (math.cos(angle) * distance),
            point2.y + (math.sin(angle) * distance),
        ]
        return Point(p1n), Point(p2n)

    def create_lane(self, base_map, line_id):
        lane = base_map.lane.add()
        lane.id.id = str(line_id)
        lane.type = map_lane_pb2.Lane.CITY_DRIVING
        lane.direction = map_lane_pb2.Lane.FORWARD
        lane.length = self.lane_length
        lane.speed_limit = 20.0
        lane.turn = map_lane_pb2.Lane.NO_TURN
        # lane.predecessor_id.add().id = str(line_id - 1)
        # lane.successor_id.add().id = str(line_id + 1)
        left_boundary = lane.left_boundary.curve.segment.add()
        right_boundary = lane.right_boundary.curve.segment.add()
        central = lane.central_curve.segment.add()
        central.length = self.lane_length

        left_boudary_type = lane.left_boundary.boundary_type.add()
        left_boudary_type.s = 0
        left_boudary_type.types.append(map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW)
        lane.right_boundary.length = self.lane_length

        right_boudary_type = lane.right_boundary.boundary_type.add()
        right_boudary_type.s = 0
        right_boudary_type.types.append(map_lane_pb2.LaneBoundaryType.DOTTED_YELLOW)
        lane.left_boundary.length = self.lane_length

        return lane, central, left_boundary, right_boundary


if __name__ == '__main__':
    MapGenSingleLine().main()
