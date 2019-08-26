"""S3 related utils, which are used for AWS S3 or Baidu BOS."""
#!/usr/bin/env python

import os

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.storage.bos_client as bos_client


# TODO: To be retired. Please use bos_client.BosClient.list_files() directly.
def list_files(bucket, prefix, suffix=''):
    """Get a RDD of files with given prefix and suffix."""
    aws_ak = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_sk = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if aws_ak is None or aws_sk is None:
        glog.error('No S3 credentials provided.')
        return None

    return BasePipeline.context().parallelize(
        bos_client.BosClient('bj', bucket, aws_ak, aws_sk).list_files(prefix, suffix))
