#!/usr/bin/env python
"""Context utils."""

import os


def is_local():
    return os.path.islink('/fuel/bazel-bin')


def is_cloud():
    return os.path.isdir('/opt/spark/work-dir')


def is_test():
    return bool(os.environ.get('TEST_BINARY'))
