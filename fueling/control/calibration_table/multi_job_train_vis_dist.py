#!/usr/bin/env python
"""Wrapper of vehicle calibration table training, visualization and distribution pipelines."""

from absl import app

from fueling.control.calibration_table.multi_job_data_distribution import MultiJobDataDistribution
from fueling.control.calibration_table.multi_job_result_visualization import MultiJobResultVisualization
from fueling.control.calibration_table.multi_job_train import MultiJobTrain


def main(argv):
    MultiJobTrain().__main__(argv)
    MultiJobResultVisualization().__main__(argv)
    MultiJobDataDistribution().__main__(argv)


if __name__ == '__main__':
    app.run(main)
