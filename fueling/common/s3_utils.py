"""S3 related utils, which are used for AWS S3 or Baidu BOS."""
#!/usr/bin/env python

import glob
import os
import string

from absl import flags
import boto3
import botocore.client
import botocore.exceptions
import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline


def s3_client(aws_ak=None, aws_sk=None):
    """Get S3 client."""
    if aws_ak is None or aws_sk is None:
        aws_ak = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_sk = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if aws_ak is None or aws_sk is None:
        glog.error('No S3 credentials provided.')
        return None
    return boto3.client('s3', endpoint_url='http://s3.bj.bcebos.com',
                        region_name='bj',
                        config=botocore.client.Config(signature_version='s3v4'),
                        aws_access_key_id=aws_ak,
                        aws_secret_access_key=aws_sk)

def list_objects(bucket, prefix, aws_ak=None, aws_sk=None):
    """
    Get a list of objects in format:
    {u'Key': '/path/to/file',
     u'LastModified': datetime.datetime(...),
     u'StorageClass': 'STANDARD',
     u'Size': 141022543, ...
    }
    """
    paginator = s3_client(aws_ak, aws_sk).get_paginator('list_objects')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in page_iterator:
        for obj in page.get('Contents', []):
            yield obj

def list_files(bucket, prefix, suffix='', to_abs_path=True):
    """Get a RDD of files with given prefix and suffix."""
    files = (BasePipeline.context()
        # RDD(obj_dict)
        .parallelize(list_objects(bucket, prefix))
        # RDD(file_obj_dict)
        .filter(lambda obj: not obj['Key'].endswith('/'))
        # RDD(file_key)
        .map(lambda obj: obj['Key']))
    if suffix:
        # RDD(file_path), ends with given suffix.
        files = files.filter(lambda path: path.endswith(suffix))
    if to_abs_path:
        # RDD(file_path), in absolute style.
        files = files.map(abs_path)
    # RDD(file_path)
    return files

def file_exists(bucket, remote_path, aws_ak, aws_sk):
    """Check if specified file is existing"""
    try:
        response = s3_client(aws_ak, aws_sk).get_object(Bucket=bucket, Key=remote_path)
    except botocore.exceptions.ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            return False
        raise ex
    return True

def upload_file(bucket, local_path, remote_path, aws_keys, meta_data={'User': 'apollo-user'}):
    """Upload a file from local to BOS"""
    aws_ak, aws_sk = aws_keys
    # arguments validation
    if not os.path.exists(local_path):
        raise ValueError('No local file/folder found for uploading')

    allowed_chars = set(string.ascii_letters + string.digits + '/' + '-' + '_')
    if set(remote_path) > allowed_chars:
        raise ValueError('Only ascii digits dash and underscore characters are allowed in paths')

    max_path_length = 1024
    if len(remote_path) > max_path_length:
        raise ValueError('Path length exceeds the limitation')

    if remote_path.endswith('/'):
        raise ValueError('Do not support uploading a folder')

    sub_paths = remote_path.split('/')
    if remote_path.startswith('/') or not file_exists(bucket, sub_paths[0] + '/', aws_ak, aws_sk):
        raise ValueError('Creating folders or files under root is not allowed')

    minimal_path_depth = 2
    maximal_path_depth = 10
    minimal_sub_path_len = 1
    maximal_sub_path_len = 256
    if (len(sub_paths) < minimal_path_depth or
        len(sub_paths) > maximal_path_depth or
        any(len(x) > maximal_sub_path_len or len(x) < minimal_sub_path_len for x in sub_paths)):
        raise ValueError('Destination path is either too short or too long')

    overwrite_whitelist = ('modules/control/control_conf/mkz7/',)
    if (file_exists(bucket, remote_path, aws_ak, aws_sk) and
        not any(remote_path.startswith(x) for x in overwrite_whitelist)):
        raise ValueError('Destination already exists')

    # Actually upload
    s3_client(aws_ak, aws_sk).upload_file(local_path, bucket, remote_path,
                                          ExtraArgs={"Metadata": meta_data})

def download_file(bucket, remote_path, local_path, aws_keys):
    """Download a file from BOS to local"""
    # Support download a dir with all files under it
    aws_ak, aws_sk = aws_keys
    if os.path.isdir(local_path):
        for obj in list_objects(bucket, remote_path, aws_ak, aws_sk):
            remote_file = obj['Key']
            # aws_skip the folder itself
            if remote_file == remote_path or remote_file == remote_path + '/':
                continue
            local_file = os.path.join(local_path, os.path.basename(remote_file))
            s3_client(aws_ak, aws_sk).download_file(bucket, remote_file, local_file)
    else:
        s3_client(aws_ak, aws_sk).download_file(bucket, remote_path, local_path)
