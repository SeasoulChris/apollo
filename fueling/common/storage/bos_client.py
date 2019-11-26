#!/usr/bin/env python
"""Baidu BOS utils."""

import os
import string

import boto3
import botocore.client
import botocore.exceptions

from fueling.common.storage.base_storage import BaseStorage
import fueling.common.logging as logging


# Constants
BOS_MOUNT_PATH = '/mnt/bos'
PARTNER_BOS_MOUNT_PATH = '/mnt/partner'


class BosClient(BaseStorage):
    """A Baidu BOS client."""

    def __init__(self, is_partner=False):
        if is_partner:
            BaseStorage.__init__(self, PARTNER_BOS_MOUNT_PATH)
            self.region = os.environ.get('PARTNER_BOS_REGION')
            self.bucket = os.environ.get('PARTNER_BOS_BUCKET')
            self.access_key = os.environ.get('PARTNER_BOS_ACCESS')
            self.secret_key = os.environ.get('PARTNER_BOS_SECRET')
        else:
            BaseStorage.__init__(self, BOS_MOUNT_PATH)
            self.region = os.environ.get('BOS_REGION')
            self.bucket = os.environ.get('BOS_BUCKET')
            self.access_key = os.environ.get('BOS_ACCESS')
            self.secret_key = os.environ.get('BOS_SECRET')

        if not self.access_key or not self.secret_key or not self.bucket or not self.region:
            logging.error('Failed to get BOS config.')
            return None

    # Override
    def list_keys(self, prefix):
        """
        Get a list of files with given prefix and suffix.
        Return absolute paths if to_abs_path is True else keys.
        """
        return [obj['Key'] for obj in self.list_objects(prefix) if not obj['Key'].endswith('/')]

    def client(self):
        """Get a boto3 client."""
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

    def file_exists(self, remote_path):
        """Check if specified file is existing"""
        try:
            self.client().get_object(Bucket=self.bucket, Key=remote_path)
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
            raise ValueError(
                'Only ascii digits dash and underscore characters are allowed in paths')

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
