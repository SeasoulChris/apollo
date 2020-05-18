#!/usr/bin/env python
"""AFS data related configuration."""

import os

# Projects
KINGLONG = "KingLong"
PILOT = "Pilot"

PROJ_TO_TABLE = {
    # Tables for "scan", "record", "keydata", "log" respectively
    KINGLONG: (
        'kinglong/auto_car/cyberecord',
        'kinglong/auto_car',
        'kinglong/auto_car/task_keydata',
        '/kinglong/auto_car'),
    PILOT: (
        'auto_car/cyberecord',
        'auto_car',
        'auto_car/task_keydata',
        '/operation/auto_car'),
}

# Maps (capture place/areas)
MAP_KL_BAIDU_DASHA = 'JinLongBaiduDaSha'
MAP_KL_BAIDU_DASHA_CESHI = 'FengTian-BaiDuDaShaCeShi'
MAP_KL_XIONGAN_ZHONGXIN = 'XiongAnShiMinZhongXin-Apollo'
MAP_KL_XIAMEN_APOLLO = 'XiaMen-Apollo'
MAP_KL_XIAMEN_RUANJIANYUAN = 'xiamenruanjianyuan-jinlong'

# Regions (not real region, but a general place that covers mutiple maps)
REGION_KL_BAIDUDASHA = 'BaiduDaSha'
REGION_KL_XIONGAN = 'XiongAn'
REGION_KL_XIAMEN = 'XiaMen'

MAP_TO_REGION = {
    MAP_KL_BAIDU_DASHA: REGION_KL_BAIDUDASHA,
    MAP_KL_BAIDU_DASHA_CESHI: REGION_KL_BAIDUDASHA,
    MAP_KL_XIONGAN_ZHONGXIN: REGION_KL_XIONGAN,
    MAP_KL_XIAMEN_APOLLO: REGION_KL_XIAMEN,
    MAP_KL_XIAMEN_RUANJIANYUAN: REGION_KL_XIAMEN,
}

# Task purposes
# http://wiki.baidu.com/pages/viewpage.action?pageId=545138637
# Skip getting data from the ones that are commented out
TASK_TO_PURPOSE = {
    0: 'debug',
    1: 'ads',
    2: 'collection',
    3: 'dailybuild',
    4: 'roadtest',
#    5: 'calibration',
    6: 'operation',
#    7: 'mapcollection',
    8: 'prerelease',
    9: 'prepublish',
    10: 'publish',
#    11: 'mapchecking'
}

# Topics that can be skipped by default
SKIP_TOPICS = ['PointCloud', 'camera']

# Logs names we need to retrieve
LOG_NAMES = 'planning.log,prediction.log'

# Target paths and paths after on the fly conversions
TARGET_PATH = 'modules/data/afs-data'
CONVERT_PATH = 'modules/data/afs-instant-data'
LOG_PATH = 'modules/data/afs-logs'


def form_target_path(target_dir, task_id, project, map_id, task_purpose):
    """Formed target path will be like target_dir/project/region/map_id/task_purpose/year/task_id"""
    return os.path.join(target_dir,
                        project,
                        MAP_TO_REGION.get(map_id, 'other'),
                        map_id,
                        TASK_TO_PURPOSE.get(int(task_purpose), task_purpose),
                        task_id.split('_')[1][:4], # Something like MKZ167_20200121131624
                        task_id)

