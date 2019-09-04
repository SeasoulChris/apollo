#!/usr/bin/env python
"""Partners."""

import collections


Partner = collections.namedtuple('Partner', ['name', 'email'])

partners = {
    'apollo': Partner(name='Apollo', email='xiaoxiangquan@baidu.com'),
    'apollo-evangelist': Partner(name='Evangelist', email='machao20@baidu.com'),
    'apollo-qa': Partner(name='QA', email='taoshengzhao01@baidu.com'),
    'udelv2019': Partner(name='Udelv', email='xiaoxiangquan@baidu.com'),
}
