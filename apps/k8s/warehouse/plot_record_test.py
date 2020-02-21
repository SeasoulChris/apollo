#!/usr/bin/env python3
# -*- coding: UTF-8-*-
"""Unit test of display_util:plot_record()."""
import sys
import webbrowser

from apps.k8s.warehouse.display_util import plot_record
from fueling.data.record_parser import RecordParser


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: plot_record_test.py <a.record>')
        sys.exit(1)
    record_meta = RecordParser.Parse(sys.argv[-1])
    html = plot_record(record_meta)

    output_html = '/tmp/plot_record_test.html'
    with open(output_html, 'w') as fout:
        fout.write('<html><body>%s</body></html>' % html)
    webbrowser.open('file://%s' % output_html)
