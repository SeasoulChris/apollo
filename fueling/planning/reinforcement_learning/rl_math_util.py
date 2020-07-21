import math


def NormalizeAngle(angle):
    a = (angle + math.pi) % (2.0 * math.pi)
    if a < 0.0:
        a += (2.0 * math.pi)
    return a - math.pi
