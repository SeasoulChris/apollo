from absl.testing import absltest
from fueling.common.base_pipeline import BasePipelineTest
from fueling.perception.pointpillars.pipelines.create_kitti_data_pipeline import CreateDataKitti

class CreateDataKittiTest(BasePipelineTest):
    def setUp(self):
        super().setUp(CreateDataKitti())
    
    def test_create_data(self):
        self.pipeline.run_test()