#!/usr/bin/env python
import glob
import operator
import os

import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class MatchImgLabel(BasePipeline):
    """Records to MatchImgLabel proto pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'match-img-label')

    def run_test(self):
        """Run test."""
        # RDD(png_img)
        png_img_rdd = self.to_rdd(glob.glob('/apollo/data/prediction/junction_img/*/*.png'))
        self.run(png_img_rdd)

    def run_prod(self):
        """Run prod."""
        source_prefix = 'modules/prediction/junction_img/'
        # RDD(png_img)
        png_img_rdd = self.to_rdd(self.bos().list_files(source_prefix, '.png'))
        self.run(png_img_rdd)

    def run(self, png_img_rdd):
        """Run the pipeline with given arguments."""
        # RDD(0/1), 1 for success
        result = png_img_rdd.map(self.process_file).cache()
        logging.info('Keeping {}/{} imgs'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_file(src_file):
        """Call prediction python code to generate labels."""
        try:
            key = os.path.basename(src_file).replace(".png", "")
            junction_label_dict = np.load(os.path.join(
                os.path.dirname(src_file).replace("junction_img", "labels"),
                'junction_label.npy')).item()
            sample_label = junction_label_dict[key]
            if len(sample_label) == 24:
                logging.info("Keeping image: " + src_file)
                return 1
        except BaseException:
            pass
        logging.info("Removing image: " + src_file)
        os.remove(src_file)
        return 0


if __name__ == '__main__':
    MatchImgLabel().main()
