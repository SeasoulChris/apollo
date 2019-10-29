#!/usr/bin/env python


class FrameCount:
    def __init__(self):
        self.count = 0

    def put(self, adc_trajectory):
        self.count += 1

    def get(self):
        frame_count = dict()
        frame_count["total"] = self.count
        return frame_count
