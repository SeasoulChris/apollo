"""
This script can filter data
"""


class Filters(object):
    """
    filter
    """

    def __init__(self, num):
        """
        init function
        """
        self.curr_index = 0
        self.num = num
        self.container = [0 for i in range(num)]

    def mean_filter(self, data):
        """
        mean filter function
        """
        index = self.curr_index % self.num
        self.container[index] = data
        self.curr_index += 1
        if self.curr_index <= self.num:
            mean = sum(self.container) / self.curr_index
        else:
            mean = sum(self.container) / self.num
        return mean
