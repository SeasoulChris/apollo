"""S3 related utils, which are used for AWS S3 or Baidu BOS."""
#!/usr/bin/env python

import os

import boto3
import botocore.client
import glog

import fueling.common.spark_utils as spark_utils


S3MountPath = '/mnt/bos'


def S3Client():
    """Get S3 client."""
    if len(os.environ.get('AWS_ACCESS_KEY_ID', '')) == 0:
        glog.error('No S3 credentials found in system environment.')
        return None
    return boto3.client('s3', endpoint_url='http://s3.bj.bcebos.com',
                        region_name='bj',
                        config=botocore.client.Config(signature_version='s3v4'))

def AbsPath(obj_rel_path):
    return os.path.join(S3MountPath, obj_rel_path)

def ListObjects(bucket, prefix=''):
    """
    Get a list of objects in format:
    {u'Key': '/path/to/file',
     u'LastModified': datetime.datetime(...),
     u'StorageClass': 'STANDARD',
     u'Size': 141022543, ...
    }
    """
    s3 = S3Client()
    paginator = s3.get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in page_iterator:
        for obj in page.get('Contents', []):
            yield obj

def ListFiles(bucket, prefix=''):
    """Get a RDD of files."""
    return spark_utils.get_context() \
        .parallelize(ListObjects(bucket, prefix)) \
        .filter(lambda obj: not obj['Key'].endswith('/')) \
        .map(lambda obj: obj['Key'])

def ListDirs(bucket, prefix=''):
    """Get a RDD of dirs."""
    return spark_utils.get_context() \
        .parallelize(ListObjects(bucket, prefix)) \
        .filter(lambda obj: obj['Key'].endswith('/')) \
        .map(lambda obj: obj['Key'])
