#!/usr/bin/env python
""" Bayesian optimization visualization related utils """

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np


def posterior(optimizer, x_obs, y_obs, grid):
    """Predict the mean and variance based on the known data"""
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, param_name, param_min, param_max, utility_function):
    """Plot the single-param Gaussian Process and Utility function"""
    param_grid = np.linspace(param_min, param_max, 10000).reshape(-1, 1)

    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        f'Gaussian Process and Utility Function After {steps} Steps',
        fontdict={'size':30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"][param_name]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, param_grid)
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(param_grid, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([param_grid, param_grid[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((param_min, param_max))
    axis.set_ylim((None, None))
    axis.set_ylabel('Target', fontdict={'size':20})
    axis.set_xlabel(param_name, fontdict={'size':20})

    utility = utility_function.utility(param_grid, optimizer._gp, 0)
    acq.plot(param_grid, utility, label='Utility Function', color='purple')
    acq.plot(param_grid[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k',
             markeredgewidth=1)
    acq.set_xlim((param_min, param_max))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel(param_name, fontdict={'size':20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    plt.show(block=False)
    plt.pause(8)
    plt.close()
