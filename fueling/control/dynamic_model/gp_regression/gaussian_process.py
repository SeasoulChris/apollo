#!/usr/bin/env python

from pyro.contrib.gp.parameterized import Parameterized

# assert pyro.__version__.startswith('1.0.0')


class GaussianProcess(Parameterized):
    """Gaussian process"""
    name = 'GaussianProcess'

    def __init__(self, args, gp_f, dataset):
        """GP initialization"""
        super(GaussianProcess, self).__init__()
        self.gp_f = gp_f
        self.dataset = dataset

    def model(self):
        """Gaussian process model"""
        return self.gp_f.model()

    def guide(self):
        """Gaussian process guide"""
        return self.gp_f.guide()

    def set_data(self, feature, label):
        """Set data for Gaussian process model"""
        # TODO(Jiaxuan): Implement proper feature setting
        self.gp_f.set_data(feature, label)

    def forward(self, input_data):
        """Call the VariantionalSparseGP with full_cov"""
        return self.gp_f.forward(input_data, full_cov=True)
