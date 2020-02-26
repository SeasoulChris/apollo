from cyber.proto import record_pb2
from cyber_py3 import record

import fueling.common.record.kinglong.cybertron.python.bag as bag
import fueling.common.record.kinglong.proto.modules.localization_pose_pb2 as localization_pose_pb2
import fueling.common.record.kinglong.proto.modules.perception_obstacle_pb2 as perception_obstacle_pb2

import modules.common.proto.geometry_pb2 as apollo_geometry_pb2
import modules.localization.proto.pose_pb2 as apollo_pose_pb2
import modules.localization.proto.localization_pb2 as apollo_localization_pb2
import modules.perception.proto.perception_obstacle_pb2 as apollo_perception_obstacle_pb2


def transfer_localization_estimate(loc):
    apollo_loc = apollo_localization_pb2.LocalizationEstimate()

    apollo_loc.pose.position.x = loc.pose.position.x
    apollo_loc.pose.position.y = loc.pose.position.y
    apollo_loc.pose.position.z = loc.pose.position.z

    apollo_loc.pose.orientation.qw = loc.pose.orientation.qw
    apollo_loc.pose.orientation.qx = loc.pose.orientation.qx
    apollo_loc.pose.orientation.qy = loc.pose.orientation.qy
    apollo_loc.pose.orientation.qz = loc.pose.orientation.qz

    apollo_loc.pose.linear_velocity.x = loc.pose.linear_velocity.x
    apollo_loc.pose.linear_velocity.y = loc.pose.linear_velocity.y
    apollo_loc.pose.linear_velocity.z = loc.pose.linear_velocity.z

    return apollo_loc


def transfer_perception_obstacles(obstacles):
    apollo_perception_obstacles = apollo_perception_obstacle_pb2.PerceptionObstacles()
    for obstacle in obstacles:
        apollo_obstacle = apollo_perception_obstacle_pb2.PerceptionObstacle()
        apollo_obstacle.id = obstacle.id
        apollo_obstacle.position.x = obstacle.position.x
        apollo_obstacle.position.y = obstacle.position.y
        apollo_obstacle.position.z = obstacle.position.z

        apollo_obstacle.theta = obstacle.theta

        apollo_obstacle.velocity.x = obstacle.velocity.x
        apollo_obstacle.velocity.y = obstacle.velocity.y
        apollo_obstacle.velocity.z = obstacle.velocity.z

        apollo_obstacle.length = obstacle.length
        apollo_obstacle.width = obstacle.width
        apollo_obstacle.height = obstacle.height

        for point in obstacle.polygon_point:
            apollo_point = apollo_geometry_pb2.Point3D()
            apollo_point.x = point.x
            apollo_point.y = point.y
            apollo_point.z = point.z
            apollo_obstacle.polygon_point.add().CopyFrom(apollo_point)

        if obstacle.type == perception_obstacle_pb2.PerceptionObstacle.UNKNOWN:
            apollo_obstacle.type = apollo_perception_obstacle_pb2.PerceptionObstacle.UNKNOWN
        elif obstacle.type == perception_obstacle_pb2.PerceptionObstacle.UNKNOWN_MOVABLE:
            apollo_obstacle.type = apollo_perception_obstacle_pb2.PerceptionObstacle.UNKNOWN_MOVABLE
        elif obstacle.type == perception_obstacle_pb2.PerceptionObstacle.UNKNOWN_UNMOVABLE:
            apollo_obstacle.type = apollo_perception_obstacle_pb2.PerceptionObstacle.UNKNOWN_UNMOVABLE
        elif obstacle.type == perception_obstacle_pb2.PerceptionObstacle.PEDESTRIAN:
            apollo_obstacle.type = apollo_perception_obstacle_pb2.PerceptionObstacle.PEDESTRIAN
        elif obstacle.type == perception_obstacle_pb2.PerceptionObstacle.BICYCLE:
            apollo_obstacle.type = apollo_perception_obstacle_pb2.PerceptionObstacle.BICYCLE
        elif obstacle.type == perception_obstacle_pb2.PerceptionObstacle.VEHICLE:
            apollo_obstacle.type = apollo_perception_obstacle_pb2.PerceptionObstacle.VEHICLE

        apollo_perception_obstacles.perception_obstacle.append(apollo_obstacle)

    return apollo_perception_obstacles


def convert_kinglong_to_apollo(kinglong_input_path, apollo_output_path):
    freader = bag.Bag(kinglong_input_path, False, False)
    topics = ['/perception/obstacles', \
              '/perception/traffic_lights', \
              '/localization/100hz/localization_pose', \
              '/pnc/planning']

    fwriter = record.RecordWriter(0, 0)
    fwriter.open(apollo_output_path)

    apollo_pose_topic = "/apollo/localization/pose"
    apollo_percption_obstacle_topic = "/apollo/perception/obstacles"

    fwriter.write_channel(apollo_pose_topic,
                          "apollo.localization.LocalizationEstimate",
                          "some descriptions")
    fwriter.write_channel(apollo_percption_obstacle_topic,
                          "apollo.perception.PerceptionObstacles",
                          "some descriptions")
    for topic, msg, msgtype, timestamp in freader.read_messages(topics):
        if topic == "/localization/100hz/localization_pose":
            loc = localization_pose_pb2.LocalizationEstimate()
            loc.ParseFromString(msg.encode('utf-8', 'surrogateescape'))
            apollo_loc = transfer_localization_estimate(loc)
            fwriter.write_message(apollo_pose_topic, apollo_loc.SerializeToString(), timestamp)
        elif topic == "/perception/obstacles":
            perception_obstacles = perception_obstacle_pb2.PerceptionObstacles()
            perception_obstacles.ParseFromString(msg.encode('utf-8', 'surrogateescape'))
            apollo_obstacles = transfer_perception_obstacles(perception_obstacles.perception_obstacle)
            fwriter.write_message(apollo_percption_obstacle_topic,
                                  apollo_obstacles.SerializeToString(), timestamp)
    freader.close()
    fwriter.close()

if __name__ == "__main__":
    convert_kinglong_to_apollo("/fuel/kl.record", "/fuel/apollo.record")
