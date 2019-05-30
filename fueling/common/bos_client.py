"""Baidu BOS utils."""
#!/usr/bin/env python

import os
import shutil
import tempfile

from absl import flags
import boto3
import botocore.client
import botocore.exceptions
import colored_glog as glog


# Configs.
flags.DEFINE_string('bos_bucket', 'apollo-platform', 'BOS bucket.')
flags.DEFINE_string('bos_region', 'bj', 'BOS region.')

# Constants
BOS_MOUNT_PATH = '/mnt/bos'
PARTNER_BOS_MOUNT_PATH = '/mnt/partner'

# Helpers.
abs_path = lambda object_key: os.path.join(BOS_MOUNT_PATH, object_key)
partner_abs_path = lambda object_key: os.path.join(PARTNER_BOS_MOUNT_PATH, object_key)


class AutoDownload(object):
    """
    Usage:
    with AutoDownload(client, bucket, key) as local_file:
        # Read local_file, which is only accessiable within the scope, as it will be removed later.
    """

    def __init__(self, boto3_client, bucket, key):
        self.boto3_client = boto3_client
        self.bucket = bucket
        self.key = key
        self.temp_dir = tempfile.mkdtemp()

    def __enter__(self):
        """Download the file as local temporary file for access."""
        local_file = os.path.join(self.temp_dir, os.path.basename(self.key))
        glog.info('Downloading key {} to {}'.format(self.key, local_file))
        try:
            self.boto3_client.download_file(self.bucket, self.key, local_file)
        except:
            glog.error('Failed to download {}, use its botfs path.'.format(self.key))
            return abs_path(self.key)
        return local_file

    def __exit__(self, type, value, traceback):
        """Remove the local temporary file."""
        shutil.rmtree(self.temp_dir)


class BosClient(object):
    """A BOS client."""
    def __init__(self, region, bucket, ak, sk):
        self.region = region
        self.bucket = bucket
        self.access_key = ak
        self.secret_key = sk
        if not self.access_key or not self.secret_key or not self.bucket or not self.region:
            glog.error('Failed to get BOS config.')
            return None

    def client(self):
        return boto3.client('s3',
                            endpoint_url='http://s3.{}.bcebos.com'.format(self.region),
                            region_name=self.region,
                            config=botocore.client.Config(signature_version='s3v4'),
                            aws_access_key_id=self.access_key,
                            aws_secret_access_key=self.secret_key)

    def list_objects(self, prefix):
        """
        Get a list of objects in format:
        {u'Key': '/path/to/file',
        u'LastModified': datetime.datetime(...),
        u'StorageClass': 'STANDARD',
        u'Size': 141022543, ...
        }
        """
        paginator = self.client().get_paginator('list_objects')
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        for page in page_iterator:
            for obj in page.get('Contents', []):
                yield obj

    def list_files(self, prefix, suffix='', to_abs_path=True):
        """Get a RDD of files with given prefix and suffix."""
        files = [obj['Key'] for obj in self.list_objects(prefix) if not obj['Key'].endswith('/')]
        if suffix:
            files = [path for path in files if path.endswith(suffix)]
        if to_abs_path:
            files = map(abs_path, files)
        return files

    def list_dirs(self, prefix, to_abs_path=True):
        """Get a RDD of dirs with given prefix."""
        dirs = [obj['Key'][:-1] for obj in self.list_objects(prefix) if obj['Key'].endswith('/')]
        return map(abs_path, dirs) if to_abs_path else dirs

    def file_exists(self, remote_path):
        """Check if specified file is existing"""
        try:
            response = self.client().get_object(Bucket=self.bucket, Key=remote_path)
        except botocore.exceptions.ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise ex
        return True

    def auto_download(self, key_or_path):
        """Return an AutoDownload instance."""
        key = key_or_path
        if key_or_path.startswith(BOS_MOUNT_PATH):
            key = key_or_path[len(BOS_MOUNT_PATH) + 1:]
        return AutoDownload(self.client(), self.bucket, key)
