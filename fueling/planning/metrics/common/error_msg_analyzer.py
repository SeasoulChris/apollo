#!/usr/bin/env python

from modules.common.proto import error_code_pb2
from fueling.planning.metrics.common.statistical_analyzer import PrintColors
from fueling.planning.metrics.common.distribution_analyzer import DistributionAnalyzer


class ErrorMsgAnalyzer:
    """class"""

    def __init__(self):
        """init"""
        self.error_msg_count = {}

    def put(self, error_msg):
        """put"""
        if len(error_msg) == 0:
            return
        if error_msg not in self.error_msg_count:
            self.error_msg_count[error_msg] = 1
        else:
            self.error_msg_count[error_msg] += 1

    def print_results(self):
        """print"""
        DistributionAnalyzer().print_distribution_results(self.error_msg_count)
