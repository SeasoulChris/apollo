#!/usr/bin/env python
"""Coord util."""
import pyproj


class CoortUtils(object):
    UTM_ZONE_ID = 10
    PROJECTOR = pyproj.Proj(proj='utm', zone=UTM_ZONE_ID, ellps='WGS84')

    @classmethod
    def utm_to_latlon(cls, x, y, utm_zone_id=None):
        """Convert UTM (x, y) to (lat, lon)."""
        projector = cls.PROJECTOR
        if utm_zone_id is not None and utm_zone_id != cls.UTM_ZONE_ID:
            projector = pyproj.Proj(proj='utm', zone=UTM_ZONE_ID, ellps='WGS84')
        return projector(x, y)
