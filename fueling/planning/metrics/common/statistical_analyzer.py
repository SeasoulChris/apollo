#!/usr/bin/env python

import numpy as np


class PrintColors:
    """ output color schema"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class StatisticalAnalyzer:
    """statistical analzer class"""

    def print_statistical_results(self, data):
        """ statistical analyzer"""
        if len(data) == 0:
            print(PrintColors.FAIL + "No Data Generated!" + PrintColors.ENDC)
            return

        arr = np.array(data)

        v = np.average(arr)
        print(PrintColors.OKBLUE + "Average: \t" + PrintColors.ENDC, "{0:.2f}".format(v))

        std = np.std(arr)
        print(PrintColors.OKBLUE + "STD: \t\t" + PrintColors.ENDC, "{0:.2f}".format(std))

        p = np.percentile(arr, 10)
        print(PrintColors.OKBLUE + "10 Percentile: \t" + PrintColors.ENDC, "{0:.2f}".format(p))

        p = np.percentile(arr, 50)
        print(PrintColors.OKBLUE + "50 Percentile: \t" + PrintColors.ENDC, "{0:.2f}".format(p))

        p = np.percentile(arr, 90)
        print(PrintColors.OKBLUE + "90 Percentile: \t" + PrintColors.ENDC, "{0:.2f}".format(p))

        p = np.percentile(arr, 99)
        print(PrintColors.OKBLUE + "99 Percentile: \t" + PrintColors.ENDC, "{0:.2f}".format(p))

        p = np.min(arr)
        print(PrintColors.OKBLUE + "min: \t" + PrintColors.ENDC, "{0:.2f}".format(p))

        p = np.max(arr)
        print(PrintColors.OKBLUE + "max: \t" + PrintColors.ENDC, "{0:.2f}".format(p))
