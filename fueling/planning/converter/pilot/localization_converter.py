#!/usr/bin/env python

from py_proto.modules.localization.proto.localization_pb2 import LocalizationEstimate


class LocalizationConverter:
    def __init__(self):
        pass

    def load(self, polit_localization_bin):
        localization = LocalizationEstimate()
        localization.ParseFromString(polit_localization_bin)
        return localization
