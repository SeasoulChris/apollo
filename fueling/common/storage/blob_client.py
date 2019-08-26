"""Azure Blob utils."""
#!/usr/bin/env python

from fueling.common.storage.base_object_storage_client import BaseObjectStorageClient


class BlobClient(BaseObjectStorageClient):
    """
    An Azure Blob client.
    Refer doc at https://azure-storage.readthedocs.io
    """

    def __init__(self, mnt_path='/mnt/blob'):
        BaseObjectStorageClient.__init__(self, mnt_path)

    # Override
    def list_keys(self, prefix):
        """
        Get a list of files with given prefix and suffix.
        Return absolute paths if to_abs_path is True else keys.
        """
        # TODO(xiaoxq): Implement.
        return []
