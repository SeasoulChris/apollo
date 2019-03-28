#!/usr/bin/env python

"""
This script manages rules that are applied to filter frames.
Inherit the base class BaseRule to add a rule and implement the filter function
"""

class RulesChain(object):
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
            frames = rule.apply(frames)
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
        frames_in_float = [float(frame[6:-5])/(10**9) for frame in frames]
        while idx < len(frames_in_float):
            filtered_frames.append(frames[idx])
            next_idx = idx+1
            while next_idx+1 < len(frames_in_float) and \
                (abs(self._interval - (frames_in_float[next_idx+1]-frames_in_float[idx])) < \
                abs(self._interval - (frames_in_float[next_idx]-frames_in_float[idx]))):
                next_idx += 1
            idx = next_idx
        return filtered_frames

def form_chains():
    # Initialize the rules objects here, which will add themselves into the rules chain
    EvenIntervalRule()