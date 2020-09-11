#!/usr/bin/env python
"""Context utils."""

from absl.testing import absltest

import fueling.common.context_utils as context_utils


class ContextUtilsTest(absltest.TestCase):

    def test_context(self):
        self.assertTrue(context_utils.is_local())
        self.assertFalse(context_utils.is_cloud())
        self.assertTrue(context_utils.is_test())


if __name__ == '__main__':
    absltest.main()
