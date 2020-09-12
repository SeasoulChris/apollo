#!/usr/bin/env python
"""
Demonstrate RAPIDS GPU acceleration on Spark SQL/DataFrame table joining
For more details refer to https://github.com/NVIDIA/spark-rapids
"""

from absl import flags
from pyspark.sql import SQLContext
from pyspark.sql.functions import col

from fueling.common.base_pipeline import BasePipeline


flags.DEFINE_integer('rows_number', 1024 * 1024, 'rows of tables that will join together')


class TableJoining(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Join two tables"""
        N = flags.FLAGS.rows_number
        SENTENCE = 'some sentence'
        data = [(x, SENTENCE) for x in range(1, N)]

        print(F'Table joining of two tables with {N} rows')

        sql_context = SQLContext(BasePipeline.SPARK_CONTEXT)
        df = sql_context.createDataFrame(data).toDF('label', 'sentence')
        df_as1 = df.alias('df_as1')
        df_as2 = df.alias('df_as2')

        joined_df = df_as1.join(df_as2, col('df_as1.label') == col('df_as2.label'), 'inner')
        joined_df.show()


if __name__ == '__main__':
    TableJoining().main()
