#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
import fueling.perception.sensor_calibration.calibration_multi_sensors as calibration_sensors


class SensorCalibrationPipelineTest(BasePipelineTest):
    def setUp(self):
        super().setUp(calibration_sensors.SensorCalibrationPipeline())

    def test_end_to_end(self):
        self.pipeline.run_test()


if __name__ == '__main__':
    absltest.main()
