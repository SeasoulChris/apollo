#!/usr/bin/env python

import fueling.common.logging as logging


class BaseCostJob(object):
    def submit(self, options):
        raise Exception('Implement me')

    def cancel(self):
        raise Exception('Implement me')
