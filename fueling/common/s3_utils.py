"""S3 related utils, which are used for AWS S3 or Baidu BOS."""
#!/usr/bin/env python

import os

import boto3
import botocore.client
import glog

import fueling.common.spark_utils as spark_utils


S3_MOUNT_PATH = '/mnt/bos'


def s3_client():
    """Get S3 client."""
    if not os.environ.get('AWS_ACCESS_KEY_ID'):
        glog.error('No S3 credentials found in system environment.')
        return None
    return boto3.client('s3', endpoint_url='http://s3.bj.bcebos.com',
                        region_name='bj',
                        config=botocore.client.Config(signature_version='s3v4'))

def abs_path(obj_rel_path):
    """Get absolute mounted path of an S3 object."""
    return os.path.join(S3_MOUNT_PATH, obj_rel_path)

def list_objects(bucket, prefix=''):
    """
    Get a list of objects in format:
    {u'Key': '/path/to/file',
     u'LastModified': datetime.datetime(...),
     u'StorageClass': 'STANDARD',
     u'Size': 141022543, ...
    }
    """
    paginator = s3_client().get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in page_iterator:
        for obj in page.get('Contents', []):
            yield obj

def list_files(bucket, prefix=''):
    """Get a RDD of files."""
    return spark_utils.get_context() \
        .parallelize(list_objects(bucket, prefix)) \
        .filter(lambda obj: not obj['Key'].endswith('/')) \
        .map(lambda obj: obj['Key'])

def list_dirs(bucket, prefix=''):
    """Get a RDD of dirs."""
    return spark_utils.get_context() \
        .parallelize(list_objects(bucket, prefix)) \
        .filter(lambda obj: obj['Key'].endswith('/')) \
        .map(lambda obj: obj['Key'])
