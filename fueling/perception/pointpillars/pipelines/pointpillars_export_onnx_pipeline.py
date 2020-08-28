#!/usr/bin/env python

import time
from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.pointpillars.second.pytorch.trans_onnx import trans_onnx
import fueling.common.logging as logging

flags.DEFINE_string('config_path',
                    '/mnt/bos/modules/perception/pointpillars/data/all.pp.mhead.config',
                    'training config path')
flags.DEFINE_string('model_path',
                    '/mnt/bos/modules/perception/pointpillars/model/voxelnet-58650.tckpt',
                    'pytorch model path')
flags.DEFINE_string('save_onnx_path', '/mnt/bos/modules/perception/pointpillars/onnx/',
                    'saved onnx path')


class PointPillarsExportOnnx(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.export_onnx)
        logging.info(
            'pointpillars export onnx complete in {} seconds.'.format(
                time.time() - time_start))

    def export_onnx(self, instance_id):
        """Run export onnx task"""

        config_path = self.FLAGS.get('config_path')
        model_path = self.FLAGS.get('model_path')
        save_onnx_path = self.FLAGS.get('save_onnx_path')

        trans_onnx(config_path, model_path, save_onnx_path)


if __name__ == '__main__':
    PointPillarsExportOnnx().main()
