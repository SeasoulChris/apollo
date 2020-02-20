#!/usr/bin/env python
"""Partners."""

import collections

_fields = ['name', 'email', 'bos_bucket', 'bos_region']
Partner = collections.namedtuple('Partner', _fields)
Partner.__new__.__defaults__ = (None,) * len(_fields)

partners = {
    'apollo': Partner(name='Apollo', email='xiaoxiangquan@baidu.com',
                      bos_bucket='apollo-platform', bos_region='bj'),
    'apollo-evangelist': Partner(name='Evangelist', email='machao20@baidu.com',
                                 bos_bucket='apollo-evangelist', bos_region='bj'),
    'apollo-qa': Partner(name='QA', email='fuyiqun@baidu.com',
                         bos_bucket='apollo-platform', bos_region='bj'),
    'udelv2019': Partner(name='Udelv', email='xiaoxiangquan@baidu.com'),
    'coolhigh': Partner(name='Coolhigh', email='mhchenm@coolhigh.com.cn'),
    'd-kit-htrob': Partner(name='HTROB', email='hance_htrob@163.com',
                           bos_bucket='ht-rob', bos_region='bj'),
    'd-kit-hytz': Partner(name='HYTZ', email='zhangxingtao2@jd.com',
                          bos_bucket='jdd-robot', bos_region='bj'),
    'd-kit-jzy': Partner(name='JZY', email='1714348719@qq.com',
                         bos_bucket='jzy-apollo', bos_region='su'),
}
