#!/usr/bin/env python
# -*- coding: UTF-8-*-

import numpy as np

PLANNING_LATENCY_BIN_0_20_MS = "latency_0_10_ms"
PLANNING_LATENCY_BIN_20_40_MS = "latency_20_40_ms"
PLANNING_LATENCY_BIN_40_60_MS = "latency_40_60_ms"
PLANNING_LATENCY_BIN_60_80_MS = "latency_60_80_ms"
PLANNING_LATENCY_BIN_80_100_MS = "latency_80_100_ms"
PLANNING_LATENCY_BIN_100_120_MS = "latency_100_120_ms"
PLANNING_LATENCY_BIN_120_150_MS = "latency_120_150_ms"
PLANNING_LATENCY_BIN_150_200_MS = "latency_150_200_ms"
PLANNING_LATENCY_BIN_200_UP_MS = "latency_200_up_ms"


class LatencyMetrics:

    def __init__(self):
        self.latency_list = []

    def process(self, planning_msg):
        latency_ms = planning_msg.latency_stats.total_time_ms
        self.latency_list.append(latency_ms)

    def get_min(self):
        return int(min(self.latency_list))

    def get_max(self):
        return int(max(self.latency_list))

    def get_avg(self):
        return int(np.average(self.latency_list))

    def get_hist(self):
        hist = {PLANNING_LATENCY_BIN_0_20_MS: 0,
                PLANNING_LATENCY_BIN_20_40_MS: 0,
                PLANNING_LATENCY_BIN_40_60_MS: 0,
                PLANNING_LATENCY_BIN_60_80_MS: 0,
                PLANNING_LATENCY_BIN_80_100_MS: 0,
                PLANNING_LATENCY_BIN_100_120_MS: 0,
                PLANNING_LATENCY_BIN_120_150_MS: 0,
                PLANNING_LATENCY_BIN_150_200_MS: 0,
                PLANNING_LATENCY_BIN_200_UP_MS: 0
                }

        for latency in self.latency_list:
            if latency < 20.0:
                hist[PLANNING_LATENCY_BIN_0_20_MS] += 1
            elif 20.0 <= latency < 40.0:
                hist[PLANNING_LATENCY_BIN_20_40_MS] += 1
            elif 40.0 <= latency < 60.0:
                hist[PLANNING_LATENCY_BIN_40_60_MS] += 1
            elif 60.0 <= latency < 80.0:
                hist[PLANNING_LATENCY_BIN_60_80_MS] += 1
            elif 80.0 <= latency < 100.0:
                hist[PLANNING_LATENCY_BIN_80_100_MS] += 1
            elif 100.0 <= latency < 120.0:
                hist[PLANNING_LATENCY_BIN_100_120_MS] += 1
            elif 120.0 <= latency < 150.0:
                hist[PLANNING_LATENCY_BIN_120_150_MS] += 1
            elif 150.0 <= latency < 200.0:
                hist[PLANNING_LATENCY_BIN_150_200_MS] += 1
            else:
                hist[PLANNING_LATENCY_BIN_200_UP_MS] += 1

        return hist
