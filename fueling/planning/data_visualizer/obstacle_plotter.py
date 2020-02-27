#!/usr/bin/env python3


def plot(perception_obstacle, ax, color):
    x = []
    y = []
    for point in perception_obstacle.polygon_point:
        x.append(point.x)
        y.append(point.y)
    x.append(perception_obstacle.polygon_point[0].x)
    y.append(perception_obstacle.polygon_point[0].y)
    ax.plot(x, y, color=color, ls="-", lw=0.5)
