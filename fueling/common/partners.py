#!/usr/bin/env python
"""Partners."""

import collections

_fields = ['name', 'email', 'bos_bucket', 'blob_container']
Partner = collections.namedtuple('Partner', _fields)
Partner.__new__.__defaults__ = (None,) * len(_fields)

partners = {
    'apollo': Partner(
        name='Apollo', email='xiaoxiangquan@baidu.com', bos_bucket='apollo-platform'),
    'apollo-evangelist': Partner(
        name='Evangelist', email='machao20@baidu.com', bos_bucket='apollo-evangelist'),
    'apollo-qa': Partner(
        name='QA', email='taoshengzhao01@baidu.com', bos_bucket='apollo-platform'),
    'udelv2019': Partner(name='Udelv', email='xiaoxiangquan@baidu.com'),
    'coolhigh': Partner(name='Coolhigh', email='mhchenm@coolhigh.com.cn'),
}
