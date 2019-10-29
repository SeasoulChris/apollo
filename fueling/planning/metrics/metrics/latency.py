#!/usr/bin/env python


import numpy as np


class Latency:
    def __init__(self):
        self.latency_list = []

    def put(self, adc_trajectory):
        self.latency_list.append(adc_trajectory.latency_stats.total_time_ms)

    def get(self):
        if len(self.latency_list) > 0:
            planning_latency = {
                "max" : max(self.latency_list),
                "min" : min(self.latency_list),
                "avg" : np.average(self.latency_list)
            }
        else:
            planning_latency = {
                "max" : 0.0,
                "min" : 0.0,
                "avg" : 0.0            
            }
        return planning_latency
