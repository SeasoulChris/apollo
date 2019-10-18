#!/usr/bin/env python

KEY_LAT_JERK_AV_SCORE = "lat_jerk_av"
KEY_LON_JERK_AV_SCORE = "lon_jerk_av"


def get_jerk_grid(jerk):
    grid_jerk = round(jerk)
    if grid_jerk == -0.0:
        grid_jerk = 0.0
    return grid_jerk


def get_angular_velocity_grid(av):
    grid_av = round(av * 10) / 10.0
    if grid_av == -0.0:
        grid_av = 0.0
    return grid_av
