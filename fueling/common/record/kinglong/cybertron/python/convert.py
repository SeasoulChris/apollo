import bag

import fueling.common.record.kinglong.proto.modules.localization_pose_pb2 as localization_pose_pb2
import fueling.common.record.kinglong.proto.modules.perception_obstacle_pb2 as perception_obstacle_pb2

import modules.common.proto.geometry_pb2 as apollo_geometry_pb2
import modules.perception.proto.perception_obstacle_pb2 as apollo_perception_obstacle_pb2
import modules.localization.proto.pose_pb2 as apollo_pose_pb2

def transfer_localization_pose(loc_pose):
    apollo_pose = apollo_pose_pb2.Pose()
    apollo_pose.position.x = loc_pose.position.x
    apollo_pose.position.y = loc_pose.position.y
    apollo_pose.position.z = loc_pose.position.z

    apollo_pose.orientation.qw = loc_pose.orientation.qw
    apollo_pose.orientation.qx = loc_pose.orientation.qx
    apollo_pose.orientation.qy = loc_pose.orientation.qy
    apollo_pose.orientation.qz = loc_pose.orientation.qz

    apollo_pose.linear_velocity.x = loc_pose.linear_velocity.x
    apollo_pose.linear_velocity.y = loc_pose.linear_velocity.y
    apollo_pose.linear_velocity.z = loc_pose.linear_velocity.z


def transfer_perception_obstacle(obstacle):
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

    return apollo_obstacle


if __name__ == "__main__":

    f = bag.Bag("/fuel/kl.record", False, False)
    topics = ['/perception/obstacles', \
              '/perception/traffic_lights', \
              '/localization/100hz/localization_pose', \
              '/pnc/planning']

    print("file name: %s" %(f.get_file_name()))
    print("total count: %d" %(f.get_message_count()))

    print("message start time: %ld" %(f.get_start_time()))
    print("message end time: %ld" %(f.get_end_time()))

    for topic, msg, msgtype, timestamp in f.read_messages(topics):
        print(topic)
        # print(topic, msgtype)
        if topic == "/localization/100hz/localization_pose":
            loc_pose = localization_pose_pb2.Pose()
            loc_pose.ParseFromString(msg)
            apollo_pose = transfer_localization_pose(loc_pose)
        elif topic == "/perception/obstacles":
            perception_obstacles = perception_obstacle_pb2.PerceptionObstacles()
            perception_obstacles.ParseFromString(msg)
            for obstacle in perception_obstacles.perception_obstacle:
                apollo_obstacle = transfer_perception_obstacle(obstacle)
                # print(apollo_obstacle)

    f.close()

