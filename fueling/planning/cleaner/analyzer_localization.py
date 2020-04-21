#!/usr/bin/env python

from modules.localization.proto import localization_pb2


class LocalizationAnalyzer:
    def __init__(self):
        self.localization_estimate = None

    def update(self, localization_estimate_msg):
        self.localization_estimate = localization_pb2.LocalizationEstimate()
        self.localization_estimate.ParseFromString(localization_estimate_msg.message)

    def get_last_localization_timestamp(self):
        if self.localization_estimate is None:
            return 0
        return self.localization_estimate.header.timestamp_sec

    def get_localization_estimate(self, msg):
        localization_estimate = localization_pb2.LocalizationEstimate()
        localization_estimate.ParseFromString(msg.message)
        return localization_estimate
