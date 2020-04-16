#!/usr/bin/env python
""" Bayesian optimization visualization related utils """

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import os
import time

import fueling.common.logging as logging


class BayesianOptimizationVisualUtils():
    """Basic functionality for Bayesian optimization visualization."""

    def __init__(self):
        """Initialize the optimization visualization"""
        self.figure = plt.figure(figsize=(12, 9))
        plt.show(block=False)
        self.figure.suptitle(
            f'Gaussian Process and Utility Function Initialization, Waiting ...'
        )
        plt.draw()
        plt.pause(1)

    def posterior(self, optimizer, x_obs, y_obs, grid):
        """Predict the mean and variance based on the known data"""
        optimizer._gp.fit(x_obs, y_obs)

        mu, sigma = optimizer._gp.predict(grid, return_std=True)
        return mu, sigma

    def plot_gp(self, optimizer, utility_function, pbounds, visual_storage_dir):
        """Plot the single-param Gaussian Process and Utility function"""
        param_name = list(pbounds)
        param_min = [pbounds[name][0] for name in param_name]
        param_max = [pbounds[name][1] for name in param_name]

        plt.clf()
        self.figure.suptitle(
            f'Gaussian Process and Utility Function After {len(optimizer.space)} Steps',
            fontdict={'size':30}
        )

        x_obs = np.array([[res["params"][name] for name in param_name] for res in optimizer.res])
        y_obs = np.array([res["target"] for res in optimizer.res])

        logging.debug(f"x_obs size: {x_obs.shape}; y_obs size: {y_obs.shape}")

        if len(param_name) == 1:
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            axis = plt.subplot(gs[0])
            acq = plt.subplot(gs[1])

            param_grid = np.linspace(param_min[0], param_max[0], 10000).reshape(-1, 1)
            mu, sigma = self.posterior(optimizer, x_obs, y_obs, param_grid)

            axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
            axis.plot(param_grid, mu, '--', color='k', label='Prediction')

            axis.fill(np.concatenate([param_grid, param_grid[::-1]]),
                      np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
                      alpha=.6, fc='c', ec='None', label='95% confidence interval')

            axis.set_xlim((param_min[0], param_max[0]))
            axis.set_ylim((None, None))
            axis.set_ylabel('Target', fontdict={'size':20})
            axis.set_xlabel(param_name[0], fontdict={'size':20})

            utility = utility_function.utility(param_grid, optimizer._gp, 0)
            acq.plot(param_grid, utility, label='Utility Function', color='purple')
            acq.plot(param_grid[np.argmax(utility)], np.max(utility), '*', markersize=15,
                     label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k',
                     markeredgewidth=1)
            acq.set_xlim((param_min[0], param_max[0]))
            acq.set_ylim((-np.max(utility) - 0.5, np.max(utility) + 0.5))
            acq.set_ylabel('Utility', fontdict={'size':20})
            acq.set_xlabel(param_name[0], fontdict={'size':20})

            axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
            acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        elif len(param_name) == 2:
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            fig_tgt = plt.subplot(gs[0], projection='3d')
            fig_acq = plt.subplot(gs[1])
            fig_mean = plt.subplot(gs[2])
            fig_std = plt.subplot(gs[3])
            cmap = plt.get_cmap('jet')

            param_x1 = np.linspace(param_min[0], param_max[0], 100)
            param_x2 = np.linspace(param_min[1], param_max[1], 100)
            grid_x1, grid_x2 = np.meshgrid(param_x1, param_x2)
            param_grid = np.hstack([grid_x1.flatten().reshape(-1, 1),
                                    grid_x2.flatten().reshape(-1, 1)])
            logging.debug(f"param_grid_0 size: {param_grid[:, 0].shape}; "
                          f"param_grid_1 size: {param_grid[:, 1].shape}")

            mu, sigma = self.posterior(optimizer, x_obs, y_obs, param_grid)
            utility = utility_function.utility(param_grid, optimizer._gp, 0)
            logging.debug(f"mu size: {mu.shape}; utility size: {utility.shape}")

            acq_value = np.array([[acq] for acq in utility])
            logging.debug(f"acq_value size: {acq_value.shape}")
            acq_im = fig_acq.pcolormesh(param_grid[:, 0].reshape(100, 100),
                                        param_grid[:, 1].reshape(100, 100),
                                        acq_value.reshape(100, 100), shading='nearest', cmap=cmap)
            fig_acq.plot(param_grid[np.argmax(acq_value), 0], param_grid[np.argmax(acq_value), 1],
                         'o', markersize=6, color='k')
            fig_acq.plot([param_grid[np.argmax(acq_value), 0], param_grid[np.argmax(acq_value), 0]],
                         [param_min[1], param_max[1]], '--k')
            fig_acq.plot([param_min[0], param_max[0]],
                         [param_grid[np.argmax(acq_value), 1], param_grid[np.argmax(acq_value), 1]],
                         '--k')
            plt.colorbar(acq_im, ax=fig_acq)
            fig_acq.set_xlim((param_min[0], param_max[0]))
            fig_acq.set_ylim((param_min[1], param_max[1]))
            fig_acq.set_xlabel(param_name[0].split('.')[-1], fontdict={'size':12})
            fig_acq.set_ylabel(param_name[1].split('.')[-1], fontdict={'size':12})
            fig_acq.set_title('Gaussian Process Acquisition Function')

            mean_value = np.array([[mean] for mean in mu])
            logging.debug(f"mean_value size: {mean_value.shape}")
            mean_im = fig_mean.pcolormesh(param_grid[:, 0].reshape(100, 100),
                                          param_grid[:, 1].reshape(100, 100),
                                          mean_value.reshape(100, 100), shading='nearest', cmap=cmap)
            fig_mean.plot(x_obs[:, 0], x_obs[:, 1], 'D', markersize=4, color='k')
            plt.colorbar(mean_im, ax=fig_mean)
            fig_mean.set_xlim((param_min[0], param_max[0]))
            fig_mean.set_ylim((param_min[1], param_max[1]))
            fig_mean.set_xlabel(param_name[0].split('.')[-1], fontdict={'size':12})
            fig_mean.set_ylabel(param_name[1].split('.')[-1], fontdict={'size':12})
            fig_mean.set_title('Gaussian Process Predicted Mean')

            std_value = np.array([[std] for std in sigma])
            logging.debug(f"std_value size: {std_value.shape}")
            std_im = fig_std.pcolormesh(param_grid[:, 0].reshape(100, 100),
                                        param_grid[:, 1].reshape(100, 100),
                                        std_value.reshape(100, 100), shading='nearest', cmap=cmap)
            fig_std.plot(x_obs[:, 0], x_obs[:, 1], 'D', markersize=4, color='k')
            plt.colorbar(std_im, ax=fig_std)
            fig_std.set_xlim((param_min[0], param_max[0]))
            fig_std.set_ylim((param_min[1], param_max[1]))
            fig_std.set_xlabel(param_name[0].split('.')[-1], fontdict={'size':12})
            fig_std.set_ylabel(param_name[1].split('.')[-1], fontdict={'size':12})
            fig_std.set_title('Gaussian Process Predicted Standard-Deviation')

            fig_tgt.scatter(x_obs[:, 0], x_obs[:, 1], y_obs)
            fig_tgt.set_xlim((param_min[0], param_max[0]))
            fig_tgt.set_ylim((param_min[1], param_max[1]))
            fig_tgt.set_xlabel(param_name[0].split('.')[-1], fontdict={'size':10})
            fig_tgt.set_ylabel(param_name[1].split('.')[-1], fontdict={'size':10})
            fig_tgt.set_zlabel('Target', fontdict={'size':10})
            fig_tgt.set_title('Gaussian Process Target Value')

            gs.tight_layout(self.figure, rect=[0, 0.03, 1, 0.95])

        else:
            gs = gridspec.GridSpec(len(param_name), 1)
            for i in range(len(param_name)):
                axis = plt.subplot(gs[i])
                axis.plot(x_obs[0:-1, i], y_obs[0:-1], 'D', markersize=6, color='k')
                axis.plot(x_obs[-1, i], y_obs[-1], 'D', markersize=6, color='r')
                axis.set_xlim((param_min[i], param_max[i]))
                axis.set_ylim((None, None))
                axis.set_ylabel('Target', fontdict={'size':20})
                axis.set_xlabel(param_name[i], fontdict={'size':20})

            gs.tight_layout(self.figure, rect=[0, 0.03, 1, 0.95])

        plt.draw()
        plt.pause(1)

        os.makedirs(visual_storage_dir)
        plt.savefig(f'{visual_storage_dir}/gaussian_process.png')
