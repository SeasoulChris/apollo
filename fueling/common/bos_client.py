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
# for test
# PARTNER_BOS_MOUNT_PATH = '/mnt/bos'

# Helpers.


def abs_path(object_key): return os.path.join(BOS_MOUNT_PATH, object_key)


def partner_abs_path(object_key): return os.path.join(PARTNER_BOS_MOUNT_PATH, object_key)


class BosClient(object):
    """A BOS client."""

    def __init__(self, region, bucket, ak, sk, mnt_path=BOS_MOUNT_PATH):
        self.region = region
        self.bucket = bucket
        self.access_key = ak
        self.secret_key = sk
        self.mnt_path = mnt_path
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

    def abs_path(self, key):
        return os.path.join(self.mnt_path, key)

    def list_files(self, prefix, suffix='', to_abs_path=True):
        """Get a RDD of files with given prefix and suffix."""
        files = [obj['Key'] for obj in self.list_objects(prefix) if not obj['Key'].endswith('/')]
        if suffix:
            files = [path for path in files if path.endswith(suffix)]
        if to_abs_path:
            files = map(self.abs_path, files)
        return files

    def list_dirs(self, prefix, to_abs_path=True):
        """Get a RDD of dirs with given prefix."""
        dirs = [obj['Key'][:-1] for obj in self.list_objects(prefix) if obj['Key'].endswith('/')]
        return map(self.abs_path, dirs) if to_abs_path else dirs

    def file_exists(self, remote_path):
        """Check if specified file is existing"""
        try:
            response = self.client().get_object(Bucket=self.bucket, Key=remote_path)
        except botocore.exceptions.ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise ex
        return True

    def upload_file(self, local_path, remote_path, meta_data={'User': 'apollo-user'}):
        """Upload a file from local to BOS"""
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
        if remote_path.startswith('/') or not self.file_exists(sub_paths[0] + '/'):
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
        if (self.file_exists(remote_path) and
            not any(remote_path.startswith(x) for x in overwrite_whitelist)):
            raise ValueError('Destination already exists')

        # Actually upload
        self.client().upload_file(local_path, self.bucket, remote_path,
                                  ExtraArgs={"Metadata": meta_data})

    def download_file(self, remote_path, local_path):
        """Download a file from BOS to local"""
        # Support download a dir with all files under it
        if os.path.isdir(local_path):
            for obj in self.list_objects(remote_path):
                remote_file = obj['Key']
                # aws_skip the folder itself
                if remote_file == remote_path or remote_file == remote_path + '/':
                    continue
                local_file = os.path.join(local_path, os.path.basename(remote_file))
                self.client().download_file(self.bucket, remote_file, local_file)
        else:
            self.client().download_file(self.bucket, remote_path, local_path)
