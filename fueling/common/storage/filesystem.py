#!/usr/bin/env python
"""Local flesystem storage."""

import glob
import os

from fueling.common.storage.base_storage import BaseStorage


class Filesystem(BaseStorage):
    """A local filesystem."""

    def __init__(self):
        BaseStorage.__init__(self, '/apollo/modules/data/fuel')

    # Override
    def list_keys(self, prefix):
        """
        Get a list of files with given prefix and suffix.
        Return absolute paths if to_abs_path is True else keys.
        """
        return [self.path_to_key(f)
                for f in glob.glob(self.abs_path(os.path.join(prefix, '**')), recursive=True)
                if os.path.isfile(f)]
