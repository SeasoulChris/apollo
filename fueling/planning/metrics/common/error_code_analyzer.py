#!/usr/bin/env python

from modules.common.proto import error_code_pb2
from fueling.planning.metrics.common.statistical_analyzer import PrintColors
from fueling.planning.metrics.common.distribution_analyzer import DistributionAnalyzer


class ErrorCodeAnalyzer:
    """class"""

    def __init__(self):
        """init"""
        self.error_code_count = {}

    def put(self, error_code):
        """put"""
        error_code_name = \
            error_code_pb2.ErrorCode.Name(error_code)
        if error_code_name not in self.error_code_count:
            self.error_code_count[error_code_name] = 1
        else:
            self.error_code_count[error_code_name] += 1

    def print_results(self):
        """print"""
        DistributionAnalyzer().print_distribution_results(self.error_code_count)
