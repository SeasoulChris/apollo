from absl.testing import absltest
from fueling.common.base_pipeline import BasePipelineTest
from fueling.perception.pointpillars.pipelines.pointpillars_training_pipeline import (
    PointPillarsTraining,)


class PointPillarsTrainingTest(BasePipelineTest):
    def setUp(self):
        super().setUp(PointPillarsTraining())

    def test_pointpillars_training(self):
        self.pipeline.run_test()


if __name__ == '__main__':
    absltest.main()
