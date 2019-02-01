#!/usr/bin/env python

import os

import boto3
import botocore.client
import glog


def S3Client():
    """Get S3 client."""
    if len(os.environ.get('AWS_ACCESS_KEY_ID', '')) == 0:
        glog.error('No S3 credentials found in system environment.')
        return None
    return boto3.client('s3', endpoint_url='http://s3.bj.bcebos.com',
                        region_name='bj',
                        config=botocore.client.Config(signature_version='s3v4'))

def ListObjects(bucket='apollo-platform', prefix=''):
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
        for obj in page['Contents']:
            yield obj

def ListObjectKeys(bucket='apollo-platform', prefix=''):
    """Return a list of keys matched the prefix."""
    for obj in ListObjects(bucket, prefix):
        yield obj['Key']
