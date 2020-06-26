#!/usr/bin/env python3
"""
The common module of admin_console
"""

import math


class Pagination(object):
    """
    The pagination class.
    """

    def __init__(self, nums, per_page_count=10, max_page_num=11):
        self.nums = nums  # The number of objects to display on a page
        self.per_page_count = per_page_count  # The number of obj to display
        self.max_page_num = max_page_num  # The number of page to display

    def get_max_page(self):
        """
        Get the total numbers of the page
        """
        return math.ceil(self.nums / self.per_page_count)


class CurrentPaginator(object):
    """
    The paginator of the current page.
    """

    def __init__(self, current_page, paginator):
        self.current_page = current_page  # The number of current page
        self.paginator = paginator  # The obj of pagination

    def has_previous(self):
        """
        Is there a previous page
        """
        return 1 < self.current_page <= self.paginator.get_max_page()

    def has_next(self):
        """
        Is there a next page
        """
        return 0 < self.current_page < self.paginator.get_max_page()

    def previous_page_number(self):
        """
        Get the number of the previous page
        """
        if self.has_previous():
            return self.current_page - 1

    def next_page_number(self):
        """
        Get the number of the next page
        """
        if self.has_next():
            return self.current_page + 1

    def page_nums(self):
        """
        Get the list of pages
        """
        mid_page = self.paginator.max_page_num // 2
        max_page = self.paginator.get_max_page()
        start = self.current_page - mid_page
        end = self.current_page + mid_page
        if start < 1:
            start = 1
        if end > max_page:
            end = max_page
        return range(start, end + 1)

    def get_index_content(self):
        """
        Get index of pre page content
        """
        top = self.paginator.per_page_count * (self.current_page - 1)
        bottom = self.paginator.per_page_count * self.current_page
        if top < 0:
            top = 0
        if bottom > self.paginator.nums:
            bottom = self.paginator.nums
        return top, bottom
