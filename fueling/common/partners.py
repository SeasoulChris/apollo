#!/usr/bin/env python
"""Partners."""

import collections

_fields = ['name', 'email', 'bos_bucket', 'bos_region', 'vehicle_sn']
Partner = collections.namedtuple('Partner', _fields)
Partner.__new__.__defaults__ = (None,) * len(_fields)

partners = {
    'apollo': Partner(name='Apollo', email='xiaoxiangquan@baidu.com',
                      bos_bucket='apollo-platform-fuel', bos_region='bj'),
    'apollo-evangelist': Partner(name='Evangelist', email='machao20@baidu.com',
                                 bos_bucket='apollo-platform-evangelist', bos_region='bj',
                                 vehicle_sn='CH0000000'),
    'apollo-qa': Partner(name='QA', email='fuyiqun@baidu.com',
                         bos_bucket='apollo-platform-fuel', bos_region='bj'),
    'udelv2019': Partner(name='Udelv', email='xiaoxiangquan@baidu.com'),
    'coolhigh': Partner(name='Coolhigh', email='mhchenm@coolhigh.com.cn'),
    'd-kit-hytz': Partner(name='HYTZ', email='zhangxingtao2@jd.com',
                          bos_bucket='jdd-robot', bos_region='bj', vehicle_sn='CH2019071'),
    'd-kit-jzy': Partner(name='JZY', email='1714348719@qq.com',
                         bos_bucket='jzy-apollo', bos_region='su', vehicle_sn='CH2019051'),
    'd-kit-jlqc-23': Partner(name='JLQC23', email='dyan1@jmc.com.cn',
                             bos_bucket='jmcauto', bos_region='su', vehicle_sn='CH2019023'),
    'd-kit-jlqc-44': Partner(name='JLQC44', email='dyan1@jmc.com.cn',
                             bos_bucket='jmcauto', bos_region='su', vehicle_sn='CH2019044'),
    'd-kit-bqyjy-73': Partner(name='BQYJY73', email='wanglanying@baicgroup.com.cn',
                              bos_bucket='baic-wly', bos_region='bj', vehicle_sn='CH2019073'),
    'd-kit-bqyjy-66': Partner(name='BQYJY66', email='golem_z@hotmail.com',
                              bos_bucket='cyouyou-calibration', bos_region='bj',
                              vehicle_sn='CH2019066'),
    'd-kit-sdxk-39': Partner(name='SDXK39', email='qzhenxing@xktech.com',
                             bos_bucket='sdxk-apollo', bos_region='bj', vehicle_sn='CH2019039'),
    'd-kit-szds-77': Partner(name='SZDS77', email='hummerautosys@gmail.com',
                             bos_bucket='hummerautosysbj', bos_region='bj',
                             vehicle_sn='CH2019077'),
    'd-kit-zkyzdhs-13': Partner(name='ZKYZDHS13', email='gaohongfeinn＠gmail.com',
                                bos_bucket='drl-coolhigh', bos_region='bj',
                                vehicle_sn='CH2019013'),
    'd-kit-qdsz-82': Partner(name='QDSZ82', email='1049892376@qq.com',
                             bos_bucket='guliduo', bos_region='bj', vehicle_sn='CH2019082'),
    'd-kit-chkj-70': Partner(name='CHKJ70', email='cuishan@coohigh.com.cn',
                             bos_bucket='coolhigh-test', bos_region='bj',
                             vehicle_sn='CH2019070'),
    'd-kit-nxkj-84': Partner(name='NXKJ84', email='tony.hong@tech-nx.com',
                             bos_bucket='nx-tech', bos_region='gz', vehicle_sn='CH2019084'),
    'd-kit-wfgk-72': Partner(name='WFGK72', email='wccsu1994@163.com',
                             bos_bucket='wifuapollo', bos_region='su', vehicle_sn='CH2019072'),
    'd-kit-xmjl-76': Partner(name='XMJL76', email='kingwingshome@outlook.com',
                             bos_bucket='kingpilot', bos_region='bj', vehicle_sn='CH2019076'),
    'd-kit-zjsys-83': Partner(name='ZJSYS83', email='liuyt@zhejianglab.com',
                              bos_bucket='zjlabsam', bos_region='su', vehicle_sn='CH2019083'),
    'd-kit-jcxx-93': Partner(name='JCXX93', email='zhangjc@iiisct.com',
                             bos_bucket='archiplab', bos_region='bj', vehicle_sn='CH2019093'),
    'd-kit-cqjt-36': Partner(name='CQJT36', email='16925652@qq.com',
                             bos_bucket='jiaodawurenche', bos_region='bj',
                             vehicle_sn='CH2019036'),
    'd-kit-sk-ncdx-01': Partner(name='SK-NCDX01', email='57186830@qq.com',
                                bos_bucket='ncuapollodkit', bos_region='su',
                                vehicle_sn='SK2020001'),
    'd-kit-sk-gksb-02': Partner(name='SK-GKSB02', email='yangguodong@cyber-ai.co',
                                bos_bucket='cyber-ai', bos_region='bj',
                                vehicle_sn='SK2020002'),
    'd-kit-shhc-2020003': Partner(name='SHHC03', email='haoxuan_geng@huacenav.com',
                                  bos_bucket='gps-imu', bos_region='bj',
                                  vehicle_sn='CH2020003'),
    'd-kit-bzl-2020011': Partner(name='BZL11', email='1014896847@qq.com',
                                 bos_bucket='ljq-apollo', bos_region='gz',
                                 vehicle_sn='CH2020011'),
    'd-kit-szy-2019091': Partner(name='SZY91', email='903381468@qq.com',
                                 bos_bucket='apollocar', bos_region='gz',
                                 vehicle_sn='CH2019091'),
}
