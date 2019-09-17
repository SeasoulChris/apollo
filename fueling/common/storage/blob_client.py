#!/usr/bin/env python
"""Azure Blob utils."""

import os

from absl import logging
from azure.storage.blob import BlockBlobService

from fueling.common.storage.base_object_storage_client import BaseObjectStorageClient


# Constants
BLOB_MOUNT_PATH = '/mnt/blob'


class BlobClient(BaseObjectStorageClient):
    """
    An Azure Blob client.
    Refer doc at https://azure-storage.readthedocs.io
    """

    def __init__(self):
        BaseObjectStorageClient.__init__(self, BLOB_MOUNT_PATH)
        self.account_name = os.environ.get('AZURE_STORAGE_ACCOUNT')
        self.account_key = os.environ.get('AZURE_STORAGE_ACCESS_KEY')
        self.container_name = os.environ.get('AZURE_BLOB_CONTAINER')
        if not self.account_name or not self.account_key or not self.container_name:
            logging.error('Failed to get Azure config.')
            return None

    # Override
    def list_keys(self, prefix):
        """
        Get a list of files with given prefix and suffix.
        Return absolute paths if to_abs_path is True else keys.
        """
        service = self.service()
        blobs = []
        marker = None
        while True:
            batch = service.list_blobs(self.container_name, prefix=prefix,
                                       delimiter='', marker=marker)
            blobs.extend([blob.name for blob in batch])
            if not batch.next_marker:
                break
            marker = batch.next_marker
        return blobs

    def service(self):
        """Get a BlobService instance."""
        return BlockBlobService(account_name=self.account_name, account_key=self.account_key)
