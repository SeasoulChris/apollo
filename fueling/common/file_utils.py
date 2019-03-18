"""File related utils."""
#!/usr/bin/env python

import errno
import os

def makedirs(dir_path):
    """Make directories recursively."""
    if os.path.exists(dir_path):
        return
    try:
        os.makedirs(dir_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

def touch(file_path):
    """Touch file."""
    if not os.path.exists(file_path):
        os.mknod(file_path)
