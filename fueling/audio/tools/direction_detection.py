#!/usr/bin/env python3

import numpy as np
import math

SOUND_SPEED = 343.2
MIC_DISTANCE_4 = 0.065
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc


def _get_direction(channels, sample_rate):
    best_guess = None

    MIC_GROUP_N = 2
    MIC_GROUP = [[0, 2], [1, 3]]

    tau = [0] * MIC_GROUP_N
    theta = [0] * MIC_GROUP_N
    for i, v in enumerate(MIC_GROUP):
        tau[i], _ = gcc_phat(channels[v[0]], channels[v[1]],
                             fs=sample_rate, max_tau=MAX_TDOA_4, interp=1)
        theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180 / math.pi

    if np.abs(theta[0]) < np.abs(theta[1]):
        if theta[1] > 0:
            best_guess = (theta[0] + 360) % 360
        else:
            best_guess = (180 - theta[0])
    else:
        if theta[0] < 0:
            best_guess = (theta[1] + 360) % 360
        else:
            best_guess = (180 - theta[1])

        best_guess = (best_guess + 90 + 180) % 360

    best_guess = (-best_guess + 120) % 360

    return best_guess


def get_direction(audio):
    channels = []
    for idx, channel_data in enumerate(audio.channel_data):
        if channel_data.channel_type != 2:
            continue
        channels.append(np.fromstring(channel_data.data, dtype='int16'))

    return _get_direction(channels, audio.microphone_config.sample_rate)
