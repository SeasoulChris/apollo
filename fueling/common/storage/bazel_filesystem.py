#!/usr/bin/env python
"""Local Bazel test filesystem storage."""

from fueling.common.storage.base_storage import BaseStorage
from fueling.common.storage.filesystem import Filesystem


class BazelFilesystem(Filesystem):
    """A local filesystem for Bazel test."""

    def __init__(self):
        BaseStorage.__init__(self, '/')
