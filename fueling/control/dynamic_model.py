#!/usr/bin/env python
import sys
import glog as log
import fueling.control.offline_evaluator.trajectory_visualization as trajectory_visualization

def Main():
    if len(sys.argv) < 2:
        log.error("please input actions")
        return
    if sys.argv[1] == 'trajectory_visualization':
        trajectory_visualization.visualize('20190201-135330',sys.argv[2])

if __name__ == '__main__':
    Main()
