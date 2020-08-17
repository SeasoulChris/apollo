import math

import torch


def NormalizeAngle(angle):
    a = (angle + math.pi) % (2.0 * math.pi)
    if torch.is_tensor(a):
        a = torch.where(a < 0.0,  a + (2.0 * math.pi), a)
    elif a < 0.0:
        a += (2.0 * math.pi)
    return a - math.pi
