from absl.testing import absltest
from fueling.common.base_pipeline import BasePipelineTest
from fueling.perception.pointpillars.pipelines.pointpillars_export_onnx_pipeline import PointPillarsExportOnnx

class PointPillarsExportOnnxTest(BasePipelineTest):
    def setUp(self):
        super().setUp(PointPillarsExportOnnx())
    
    def test_pointpillars_export_onnx(self):
        self.pipeline.run_test()