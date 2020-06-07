#!/usr/bin/env python

import os

from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset import \
    DynamicModelDataset
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.h5_utils as h5_utils


def check_feature_data(data_dir):
    features = None
    hdf5_files = file_utils.list_files_with_suffix(data_dir, '.hdf5')
    for idx, hdf5_file in enumerate(hdf5_files):
        logging.debug(f'hdf5_file: {hdf5_file}')
        if features is None:
            features = h5_utils.read_h5(hdf5_file)
        else:
            features = np.concatenate((features, h5_utils.read_h5(hdf5_file)), axis=0)
    logging.debug(features.shape)
    for i in range(0, features.shape[0]):
        cnt = 0
        if features[i, 14] < 0:
            cnt += 1
            logging.info(f'feature idx {idx} is {features[i,[0,14,15,16,17,22]]}')

    return features


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return ((ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
            ax_colorbar)


def plot_distribution(axes, X, y, cmap, hist_nbins=50, title="",
                      x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cmap(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_X1.axis('off')

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_X0.axis('off')


def plot_distribution_input_only(axes, X, hist_nbins=50, title="",
                                 x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_X1.axis('off')

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_X0.axis('off')


def make_plot_input_only(distributions, item_idx):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution_input_only(axarr[0], X, hist_nbins=200,
                                 x0_label="Brake",
                                 x1_label="Throttle",
                                 title="Full data")
    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = (
        np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1)
        & np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
    plot_distribution_input_only(axarr[1], X[non_outliers_mask],
                                 hist_nbins=50,
                                 x0_label="Brake",
                                 x1_label="Throttle",
                                 title="Zoom-in")


def make_plot(distributions, item_idx, y, cmap, y_full):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(axarr[0], X, y, cmap, hist_nbins=200,
                      x0_label="Brake",
                      x1_label="Throttle",
                      title="Full data")

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = (
        np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1)
        & np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
    plot_distribution(axarr[1], X[non_outliers_mask], y[non_outliers_mask], cmap,
                      hist_nbins=50,
                      x0_label="Brake",
                      x1_label="Throttle",
                      title="Zoom-in")

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap,
                              norm=norm, orientation='vertical',
                              label='Color mapping for values of y')


def visualize_input_only(features, input_indices):
    # data already exist in features
    X = features[:, input_indices]
    distributions = [
        ('Unscaled data', X),
        ('Data after standard scaling',
         StandardScaler().fit_transform(X)),
        ('Data after min-max scaling',
         MinMaxScaler().fit_transform(X)),
        ('Data after max-abs scaling',
         MaxAbsScaler().fit_transform(X)),
        ('Data after robust scaling',
         RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
        ('Data after power transformation (Yeo-Johnson)',
         PowerTransformer(method='yeo-johnson').fit_transform(X)),
        ('Data after power transformation (Box-Cox)',
         PowerTransformer(method='box-cox').fit_transform(X)),
        ('Data after quantile transformation (gaussian pdf)',
         QuantileTransformer(output_distribution='normal')
         .fit_transform(X)),
        ('Data after quantile transformation (uniform pdf)',
         QuantileTransformer(output_distribution='uniform')
         .fit_transform(X)),
        ('Data after sample-wise L2 normalizing',
         Normalizer().fit_transform(X)),
    ]
    for i in range(0, len(distributions)):
        make_plot_input_only(distributions, i)
        plt.show()


def visualize(datasets, input_indices, output_index):
    X_full = []
    y_full = []
    logging.info(input_indices)
    logging.info(output_index)
    logging.info(len(datasets))
    for i, dataset in enumerate(datasets):
        logging.info(dataset[0].shape)
        logging.info(dataset[1].shape)
        X_full.append(dataset[0][:, input_indices])
        y_full.append(dataset[1][output_index] * np.ones((100, 1)))
    # X = np.array(X_full)
    # X = np.stack(X_full, axis=0)
    # logging.info(y_full.shape)
    X = np.concatenate(X_full, axis=0)
    logging.info(X.shape)
    logging.info(np.min(X[:, 0]))
    logging.info(np.min(X[:, 1]))

    distributions = [
        ('Unscaled data', X),
        ('Data after standard scaling',
         StandardScaler().fit_transform(X)),
        ('Data after min-max scaling',
         MinMaxScaler().fit_transform(X)),
        ('Data after max-abs scaling',
         MaxAbsScaler().fit_transform(X)),
        ('Data after robust scaling',
         RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
        ('Data after power transformation (Yeo-Johnson)',
         PowerTransformer(method='yeo-johnson').fit_transform(X)),
        # ('Data after power transformation (Box-Cox)',
        #  PowerTransformer(method='box-cox').fit_transform(X)),
        ('Data after quantile transformation (gaussian pdf)',
         QuantileTransformer(output_distribution='normal')
         .fit_transform(X)),
        ('Data after quantile transformation (uniform pdf)',
         QuantileTransformer(output_distribution='uniform')
         .fit_transform(X)),
        ('Data after sample-wise L2 normalizing',
         Normalizer().fit_transform(X)),
    ]

    # scale the output between 0 and 1 for the colorbar
    # X = np.concatenate(y_full, axis=0)
    y_full_array = np.concatenate(y_full, axis=0)
    y = minmax_scale(y_full_array)
    logging.info(y.shape)

    # plasma does not exist in matplotlib < 1.5
    cmap = getattr(cm, 'plasma_r', cm.hot_r)
    for i in range(0, len(distributions)):
        make_plot(distributions, i, y, cmap, y_full_array)
        plt.show()


def check_outlier(data_dir):
    """Extract datasets from data path"""
    # list of dataset = (input_tensor, output_tensor)
    h5_files = file_utils.list_files_with_suffix(data_dir, '.h5')
    for idx, h5_file in enumerate(h5_files):
        logging.debug(f'h5_file: {h5_file}')
        with h5py.File(h5_file, 'r') as model_norms_file:
            # Get input data
            input_segment = np.array(model_norms_file.get('input_segment'))
            if np.isnan(np.sum(input_segment)):
                logging.error(f'file {h5_file} contains NAN data in input segment')
            # Get output data
            output_segment = np.array(model_norms_file.get('output_segment'))
            if np.any(np.max(output_segment)) > 1.0:
                logging.error(f'file {h5_file} contains large_label')


if __name__ == '__main__':
    # initialize obj
    dynamic_model_dataset = DynamicModelDataset(
        data_dir='/fuel/fueling/control/dynamic_model_2_0/gp_regression/testdata/train',
        is_standardize=False)
    logging.info(f'dataset length {len(dynamic_model_dataset.datasets)}')
    # what to do
    validate_result = False
    visualize_normalization_method = False
    check_data = False
    check_large_label = True
    # validation processing:
    if validate_result:
        processed_data = dynamic_model_dataset.getitem(0)[0]
        for id in range(1, len(dynamic_model_dataset.datasets)):
            processed_data = torch.cat((processed_data, dynamic_model_dataset.getitem(id)[0]), 0)
        logging.info(f'processed data shape is {processed_data.shape}')
        # visualize standardized data
        for id in range(0, 6):
            plt.figure(figsize=(12, 8))
            plt.plot(processed_data[:, id], 'b.')
            logging.info(
                f'mean value for {id} is '
                + f'{np.mean(processed_data[:, id].numpy(), dtype=np.float64)}')
            logging.info(
                f'std value for {id} is {np.std(processed_data[:, id].numpy(), dtype=np.float64)}')
            logging.info(
                f'max value for {id} is {np.amax(processed_data[:, id].numpy())}')
            logging.info(
                f'min value for {id} is {np.amin(processed_data[:, id].numpy())}')
            plt.show()
    # visualization
    if visualize_normalization_method:
        input_indices = np.array([2, 3])
        output_index = 0
        datasets = dynamic_model_dataset.datasets
        visualize(datasets, input_indices, output_index)
    if check_data:
        features = check_feature_data(
            '/fuel/fueling/control/dynamic_model_2_0/gp_regression/testdata/train')
        input_indices = np.array([15, 16])
        visualize_input_only(features, input_indices)
    if check_large_label:
        check_outlier('/fuel/fueling/control/dynamic_model_2_0/0603')
