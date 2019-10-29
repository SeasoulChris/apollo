#!/usr/bin/env python

from fueling.planning.metrics.common.statistical_analyzer import PrintColors


class DistributionAnalyzer:
    """statistical analzer class"""

    def print_distribution_results(self, data):
        """distribution analyzer"""
        if len(data) == 0:
            print(PrintColors.FAIL + "No Data Generated!" + PrintColors.ENDC)
            return

        total = sum(data.values())

        for k, v in data.items():
            percentage = "{0:.2f}".format((float(v) / total) * 100)
            print(PrintColors.OKBLUE + k + " = " + str(v) + \
                "(" + percentage + "%)" + PrintColors.ENDC)
