#!/usr/bin/env python


class HmiAnalyzer:
    def __init__(self):
        self.hmi_status = None
        self.hmi_status_msg = None

    def update(self, hmi_status_msg):
        self.hmi_status_msg = hmi_status_msg

    def get_hmi_status_msg(self):
        return self.hmi_status_msg
