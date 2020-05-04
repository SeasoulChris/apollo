#!/usr/bin/env python

import os
import time

from absl import flags
import numpy as np
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.common.h5_utils import read_h5
from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index
import fueling.common.logging as logging
import fueling.control.common.multi_vehicle_plot_utils as plot_utils

flags.DEFINE_integer('percentile', '50', 'percentile of the data.')


class DynamicModelDatasetVisualization(BasePipeline):
    """Dynamic model 2.0 dataset visualization"""

    def filter_dimensions(self, segment):
        """From segment(M, N) to segment(m, n) according to config"""
        chosen_columns = ['speed', 'throttle', 'brake', 'steering']
        chosen_column_idxs = [segment_index[column] for column in chosen_columns]
        column_percentile = np.percentile(segment[:, chosen_column_idxs],
                                          self.FLAGS.get('percentile'), axis=0)
        filtered_segment = []
        for idx, column in enumerate(chosen_columns):
            filtered_segment.append((column, column_percentile[idx]))
            logging.info(F'column {column}: percentile value {column_percentile[idx]}')
        return filtered_segment


    def run(self):
        """Run."""
        input_data_path = (self.FLAGS.get('input_data_path') or
            'modules/control/dynamic_model_2_0/features')
        output_data_path = self.our_storage().abs_path(self.FLAGS.get('output_data_path') or
            os.path.join(input_data_path, F'dataset_distrib_{time.strftime("%Y%m%d-%H%M%S")}.pdf'))

        logging.info(F'Result File: {output_data_path}')

        index_values = spark_helper.cache_and_log('IndexValues',
            # RDD(hdf5 files)
            self.to_rdd(self.our_storage().list_files(input_data_path, '.hdf5'))
            # RDD(segments), each segment is (100 x 22) as defined in current config
            .map(read_h5)
            # PairRDD(index, value), the value is a certain result aggregated from 100
            .flatMap(self.filter_dimensions)
            # PairRDD(index, (values))
            .groupByKey())

        plot_utils.plot_dynamic_mode_2_0_feature_hist(index_values.collect(), output_data_path)
        logging.info('Done with DynamicModelDatasetVisualization')


if __name__ == '__main__':
    DynamicModelDatasetVisualization().main()
