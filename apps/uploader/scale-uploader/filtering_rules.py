#!/usr/bin/env python

"""
This script manages rules that are applied to filter frames.
Inherit the base class BaseRule to add a rule and implement the filter function
"""

import math
import os

import colored_glog as glog

class RulesChain(object):
    """Rules organizor"""
    # The static list of rules that will apply one by one
    rules = []

    @staticmethod
    def register(rule):
        """Add the rule into the processing chain"""
        RulesChain.rules.append(rule)

    @staticmethod
    def do_filter(frames):
        """Filter with rules one by one"""
        for rule in RulesChain.rules:
            glog.info('filtering with rule: {}, current frames: {}'.format(rule.name, len(frames)))
            frames = rule.apply(frames)
            glog.info('after rule: {}, left frames: {}'.format(rule.name, len(frames)))
        return frames

class BaseRule(object):
    """Base class for rules"""
    def __init__(self, name):
        """Constructor"""
        self.name = name
        RulesChain.register(self)

    def apply(self, frames):
        """Apply the rule"""
        raise Exception('{}::apply() not implemented for base class'.format(self.name))

class EvenIntervalRule(BaseRule):
    """The rule to make frames as even as possible, for example one frame per 1 ms"""
    def __init__(self):
        """Constructor"""
        BaseRule.__init__(self, 'Even-Interval-Rule')
        self._interval = 0.5

    def apply(self, frames):
        """
        Loop the input frames list and look for the closest timestamp towards the given interval
        """
        idx = 0
        filtered_frames = list()
        frames_in_float = [float(os.path.basename(frame)[6:-5])/(10**9) for frame in frames]
        while idx < len(frames_in_float):
            filtered_frames.append(frames[idx])
            next_idx = idx+1
            while next_idx+1 < len(frames_in_float) and \
                (abs(self._interval - (frames_in_float[next_idx+1]-frames_in_float[idx])) < \
                abs(self._interval - (frames_in_float[next_idx]-frames_in_float[idx]))):
                next_idx += 1
            idx = next_idx
        return filtered_frames

class MovingCarRule(BaseRule):
    """The rule to filter out the frames where vehicle is standing still, by using GPS info"""
    def __init__(self):
        """Constructor"""
        BaseRule.__init__(self, 'Moving-Car-Rule')
        self._distance = 1.0
        self._pre_xpos = None
        self._pre_ypos = None

    def apply(self, frames):
        """Filter out frames if the distance is not big enough which means it's not moving"""
        filtered_frames = list()
        for frame in frames:
            gps_data = self._load_gps_from_file(frame)
            xpos = float(next(gps_info for gps_info in gps_data if gps_info.find('"x":') != -1)
                         .split(':')[1].strip(', \n'))
            ypos = float(next(gps_info for gps_info in gps_data if gps_info.find('"y":') != -1)
                         .split(':')[1].strip(', \n'))
            if self._pre_xpos and self._pre_ypos:
                if math.sqrt((xpos-self._pre_xpos)**2 + (ypos-self._pre_ypos)**2) > self._distance:
                    filtered_frames.append(frame)
            else:
                filtered_frames.append(frame)
            self._pre_xpos = xpos
            self._pre_ypos = ypos
        return filtered_frames

    def _load_gps_from_file(self, frame_file_path):
        """
        Load first N lines of text which contains GPS info from frame file,
        instead of loading the whole json file to improve efficiency
        """
        lines_number = 15
        with open(frame_file_path) as frame_file:
            return [next(frame_file) for x in range(lines_number)]

class ExactBatchSizeRule(BaseRule):
    """The rule to get an exact number of frames per task"""
    def __init__(self):
        """Constructor"""
        BaseRule.__init__(self, 'Exact-BatchSize-Rule')
        self._batch_size = 50

    def apply(self, frames):
        """Trunc the frames if it's over batch size, and returns none if less than batch size"""
        if len(frames) < self._batch_size:
            return []
        return frames[:50]

def form_chains():
    """Chain all the rules together"""
    # Initialize the rules objects here, which will add themselves into the rules chain
    EvenIntervalRule()
    MovingCarRule()
    ExactBatchSizeRule()
