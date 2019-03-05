"""File related utils."""
#!/usr/bin/env python

import errno
import os

def makedirs(dir_path):
    """Make directories recursively."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
    return dir_path

def touch(file_path):
    """Touch file."""
    os.mknod(file_path) if not os.path.exists(file_path) else None
    return file_path
