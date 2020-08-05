#!/usr/bin/env python

import os
import time

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.common.h5_utils import read_h5
import fueling.common.logging as logging
import fueling.common.spark_helper as spark_helper
import fueling.control.common.multi_vehicle_plot_utils as plot_utils
import fueling.control.dynamic_model_2_0.feature_extraction.feature_extraction_utils as \
    feature_utils

flags.DEFINE_integer('percentile', 50, 'percentile of the data.')


class DynamicModelDatasetVisualization(BasePipeline):
    """Dynamic model 2.0 dataset visualization"""

    def run(self):
        """Run."""
        input_data_path = (
            self.FLAGS.get('input_data_path')
            or 'modules/control/dynamic_model_2_0/features')
        output_data_path = self.our_storage().abs_path(
            self.FLAGS.get('output_data_path')
            or os.path.join(input_data_path,
                            F'dataset_distrib_{time.strftime("%Y%m%d-%H%M%S")}.pdf'))

        logging.info(F'Result File: {output_data_path}')

        index_values = spark_helper.cache_and_log(
            'IndexValues',
            # RDD(hdf5 files)
            self.to_rdd(self.our_storage().list_files(input_data_path, '.hdf5'))
            # RDD(segments), each segment is (100 x 22) as defined in current config
            .map(read_h5)
            # PairRDD(index, value), the value is a certain result aggregated from 100
            .flatMap(lambda x: feature_utils.filter_dimensions(x, self.FLAGS.get('percentile')))
            # PairRDD(index, (values))
            .groupByKey())

        plot_utils.plot_dynamic_mode_2_0_feature_hist(index_values.collect(), output_data_path)
        logging.info('Done with DynamicModelDatasetVisualization')


if __name__ == '__main__':
    DynamicModelDatasetVisualization().main()
