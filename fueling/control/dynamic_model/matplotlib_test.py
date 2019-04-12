#!/usr/bin/env python

import os
import glob
import time

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.s3_utils as s3_utils


class MatplotlibTest(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'dynamic_model')

    def run_test(self):
        timestr = time.strftime("%Y%m%d-%H%M%S") 
        pdf_file = \
            '/apollo/modules/data/fuel/testdata/control/learning_based_model/evaluation_result\
            Dataset_Distribution_%s.pdf' % timestr
        self.run(pdf_file)

    def run_prod(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        pdf_file = \
            '/mnt/bos/modules/control/evaluation_result/Dataset_Distribution_%s.pdf' % timestr
        self.run(pdf_file)
    
    def run(self, pdf_file):
        fearure = np.linspace(0, 10, 100)
        with PdfPages(pdf_file) as pdf:
            plt.figure(figsize=(4,3))
            plt.hist(fearure, fearure, bins ='auto', label='linear')
            plt.title ("Histogram of the Feature Input")
            plt.show()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


if __name__ == '__main__':
    MatplotlibTest().main()
