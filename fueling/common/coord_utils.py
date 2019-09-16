#!/usr/bin/env python
"""Coord util."""

import numpy as np
import pyproj


class CoordUtils(object):
    UTM_ZONE_ID = 10
    PROJECTOR = pyproj.Proj(proj='utm', zone=UTM_ZONE_ID, ellps='WGS84')

    @classmethod
    def utm_to_latlon(cls, x, y, utm_zone_id=None):
        """Convert UTM (x, y) to (lat, lon)."""
        projector = cls.PROJECTOR
        if utm_zone_id is not None and utm_zone_id != cls.UTM_ZONE_ID:
            projector = pyproj.Proj(proj='utm', zone=UTM_ZONE_ID, ellps='WGS84')
        lon, lat = projector(x, y, inverse=True)
        return (lat, lon)

    @classmethod
    def latlon_to_utm(cls, lat, lon, utm_zone_id=None):
        """Convert (lat, lon) to UTM (x, y)."""
        projector = cls.PROJECTOR
        if utm_zone_id is not None and utm_zone_id != cls.UTM_ZONE_ID:
            projector = pyproj.Proj(proj='utm', zone=UTM_ZONE_ID, ellps='WGS84')
        return projector(lon, lat)

    @staticmethod
    def world_to_relative(input_world_coord, ref_world_coord):
        """Convert world coord to relative coord."""
        x_diff = input_world_coord[0] - ref_world_coord[0]
        y_diff = input_world_coord[1] - ref_world_coord[1]
        rho = np.sqrt(x_diff ** 2 + y_diff ** 2)
        theta = np.arctan2(y_diff, x_diff) - ref_world_coord[2]
        return (np.cos(theta) * rho, np.sin(theta) * rho)
