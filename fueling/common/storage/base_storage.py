#!/usr/bin/env python
"""Common object storage client utils."""

import os


class BaseStorage(object):
    """An object storage client."""

    def __init__(self, mnt_path):
        self.mnt_path = mnt_path

    def abs_path(self, path):
        """Return absolute mounting path of the given key."""
        if path.startswith('/'):
            return path
        return os.path.join(self.mnt_path, path)

    def relative_path(self, path):
        """Remove mnt_path it is abs_path."""
        if path.startswith(self.mnt_path):
            path = path[len(self.mnt_path) + 1:]
        return path

    def list_keys(self, prefix):
        """Get a list of keys with given prefix."""
        raise Exception('{}::list_keys not implemented!'.format(self.__class__.__name__))

    def list_files(self, prefix, suffix='', to_abs_path=True):
        """
        Get a list of files with given prefix and suffix.
        Return absolute paths if to_abs_path is True else keys.
        Allow prefix to be both absolute path and relative path.
        """
        files = self.list_keys(self.relative_path(prefix))
        if suffix:
            files = [path for path in files if path.endswith(suffix)]
        if to_abs_path:
            files = list(map(self.abs_path, files))
        return files

    def list_end_dirs(self, prefix, to_abs_path=True):
        """
        Get a list of dirs with given prefix, which contain at least one file.
        Return absolute paths if to_abs_path is True else keys.
        Allow prefix to be both absolute path and relative path.
        """
        files = self.list_keys(self.relative_path(prefix))
        dirs = set(map(os.path.dirname, files))
        if to_abs_path:
            dirs = map(self.abs_path, dirs)
        return list(dirs)
