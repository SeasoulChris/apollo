#!/usr/bin/env python

import fueling.common.logging as logging


class BaseCostJob(object):
    def run(self, options):
        """Return boolean to indicate the job is done successfully or not"""
        raise Exception('Implement me')

    def cancel(self):
        raise Exception('Implement me')
