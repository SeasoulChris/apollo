#!/usr/bin/env python
"""Partners."""

import collections

_fields = ['name', 'email', 'bos_bucket', 'bos_region']
Partner = collections.namedtuple('Partner', _fields)
Partner.__new__.__defaults__ = (None,) * len(_fields)

partners = {
    'apollo': Partner(name='Apollo', email='xiaoxiangquan@baidu.com',
                      bos_bucket='apollo-platform-fuel', bos_region='bj'),
    'apollo-evangelist': Partner(name='Evangelist', email='machao20@baidu.com',
                                 bos_bucket='apollo-platform-evangelist', bos_region='bj'),
    'apollo-qa': Partner(name='QA', email='fuyiqun@baidu.com',
                         bos_bucket='apollo-platform-fuel', bos_region='bj'),
    'udelv2019': Partner(name='Udelv', email='xiaoxiangquan@baidu.com'),
    'coolhigh': Partner(name='Coolhigh', email='mhchenm@coolhigh.com.cn'),
    'd-kit-htrob': Partner(name='HTROB', email='hance_htrob@163.com',
                           bos_bucket='ht-rob', bos_region='bj'),
    'd-kit-hytz': Partner(name='HYTZ', email='zhangxingtao2@jd.com',
                          bos_bucket='jdd-robot', bos_region='bj'),
    'd-kit-jzy': Partner(name='JZY', email='1714348719@qq.com',
                         bos_bucket='jzy-apollo', bos_region='su'),
    'd-kit-jlqc-23': Partner(name='JLQC23', email='dyan1@jmc.com.cn',
                             bos_bucket='jmcauto', bos_region='su'),
    'd-kit-jlqc-44': Partner(name='JLQC44', email='dyan1@jmc.com.cn',
                             bos_bucket='jmcauto', bos_region='su'),
    'd-kit-bqyjy-73': Partner(name='BQYJY73', email='wanglanying@baicgroup.com.cn',
                              bos_bucket='baic-wly', bos_region='bj'),
    'd-kit-bqyjy-66': Partner(name='BQYJY66', email='golem_z@hotmail.com',
                              bos_bucket='cyouyou-calibration', bos_region='bj'),
    'd-kit-sdxk-39': Partner(name='SDXK39', email='qzhenxing@xktech.com',
                             bos_bucket='sdxk-apollo', bos_region='bj'),
    'd-kit-szds-77': Partner(name='SZDS77', email='hummerautosys@gmail.com',
                             bos_bucket='hummerautosysbj', bos_region='bj'),
    'd-kit-zkyzdhs-13': Partner(name='ZKYZDHS13', email='gaohongfeinn＠gmail.com',
                                bos_bucket='drl-coolhigh', bos_region='bj'),
    'd-kit-qdsz-82': Partner(name='QDSZ82', email='1049892376@qq.com',
                             bos_bucket='guliduo', bos_region='bj'),
    'd-kit-chkj-70': Partner(name='CHKJ70', email='cuishan@coohigh.com.cn',
                             bos_bucket='coolhigh-test', bos_region='bj'),
}
