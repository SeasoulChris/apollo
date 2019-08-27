"""Azure Blob utils."""
#!/usr/bin/env python

from azure.storage.blob import BlockBlobService

from fueling.common.storage.base_object_storage_client import BaseObjectStorageClient


class BlobClient(BaseObjectStorageClient):
    """
    An Azure Blob client.
    Refer doc at https://azure-storage.readthedocs.io
    """

    def __init__(self, container_name, account_name, account_key, mnt_path='/mnt/blob'):
        BaseObjectStorageClient.__init__(self, mnt_path)
        self.container_name = container_name
        self.account_name = account_name
        self.account_key = account_key

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
