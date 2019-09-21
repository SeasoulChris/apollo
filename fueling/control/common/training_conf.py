#!/usr/bin/env python

try:
    # TODO(xiaoxq): Retire.
    from sets import Set  # Python 2
except ImportError:
    Set = set  # Python 3


vehicle_list = Set(['ch9'])

inter_result_folder = 'modules/control/tmp/results'
# output_folder = 'modules/control/calibration_table/results'
output_folder = 'modules/control/data/results/CalibrationTableConf'
