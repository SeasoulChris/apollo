"""S3 related utils, which are used for AWS S3 or Baidu BOS."""
#!/usr/bin/env python

import os
import string

import boto3
import botocore.client
import glog
import pyspark_utils.helper as spark_helper


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
    return spark_helper.get_context() \
        .parallelize(list_objects(bucket, prefix)) \
        .filter(lambda obj: not obj['Key'].endswith('/')) \
        .map(lambda obj: obj['Key'])

def list_dirs(bucket, prefix=''):
    """Get a RDD of dirs."""
    return spark_helper.get_context() \
        .parallelize(list_objects(bucket, prefix)) \
        .filter(lambda obj: obj['Key'].endswith('/')) \
        .map(lambda obj: obj['Key'])

def file_exists(bucket, remote_path):
    """Check if specified file is existing"""
    return len(list(list_objects(bucket, remote_path))) > 0

def upload_file(bucket, local_path, remote_path, meta_data=None):
    """Upload a file from local to BOS"""
    # arguments validation
    if not os.path.exists(local_path):
        raise ValueError('No local file/folder found for uploading')

    allowed_chars = set(string.ascii_letters + string.digits + '/' + '-' + '_')
    if set(remote_path) > allowed_chars:
        raise ValueError('Only ascii digits dash and underscore characters are allowed in paths')

    sub_paths = remote_path.split('/')
    minimal_path_depth = 2
    maximal_path_depth = 10
    if len(sub_paths) < minimal_path_depth or len(sub_paths) > maximal_path_depth:
        raise ValueError('Destination path is either too short or too long')

    if file_exists(bucket, remote_path):
        raise ValueError('Destination already exists')

    if not file_exists(bucket, sub_paths[0]):
        raise ValueError('Creating folders or files under root is not allowed')

    # Set default MetaData if it's not specified
    if meta_data is None:
        meta_data = {'User': 'apollo-user'}

    # Actually upload
    s3_client().upload_file(local_path,
                            bucket,
                            remote_path,
                            ExtraArgs={"Metadata": meta_data})

def download_file(bucket, remote_path, local_path):
    """Download a file from BOS to local"""
    s3_client().download_file(bucket, remote_path, local_path)
