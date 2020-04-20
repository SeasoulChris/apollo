from modules.localization.proto import localization_pb2
from modules.routing.proto.routing_pb2 import RoutingResponse

from fueling.planning.cleaner.map_reader import MapReader


class AgentAnalyzer:
    def __init__(self):
        self.map_reader = MapReader()

    def process(self, msgs):
        for msg in msgs:
            if msg.topic == "/apollo/routing_response":
                routing_response = RoutingResponse()
                routing_response.ParseFromString(msg.message)

            if msg.topic == '/apollo/localization/pose':
                localization_estimate = localization_pb2.LocalizationEstimate()
                localization_estimate.ParseFromString(msg.message)
