#!/usr/bin/env python

import numpy as np

from fueling.control.dynamic_model.conf.model_config import feature_config
import fueling.common.logging as logging
import fueling.control.dynamic_model.data_generator.holistic_data_generator as \
    holistic_data_generator
import fueling.control.dynamic_model.data_generator.non_holistic_data_generator as \
    non_holistic_data_generator


# Constants
DIM_SEQUENCE_LENGTH = feature_config["sequence_length"]
DIM_DELAY_STEPS = feature_config["delay_steps"]


def generate_training_data(segment, is_holistic=False):
    """
    extract usable features from the numpy array for model training
    """
    total_len = segment.shape[0] - DIM_DELAY_STEPS
    total_sequence_num = segment.shape[0] - DIM_DELAY_STEPS - DIM_SEQUENCE_LENGTH
    logging.info('Total length: {}'.format(total_len))

    if is_holistic:
        dim_input = feature_config["holistic_input_dim"]
        dim_output = feature_config["holistic_output_dim"]
        mlp_input_data, mlp_output_data = holistic_data_generator.generate_mlp_data(
            segment, total_len)
    else:
        dim_input = feature_config["input_dim"]
        dim_output = feature_config["output_dim"]
        mlp_input_data, mlp_output_data = non_holistic_data_generator.generate_mlp_data(
            segment, total_len)

    lstm_input_data = np.zeros(
        [total_sequence_num, dim_input, DIM_SEQUENCE_LENGTH], order='C')
    lstm_output_data = np.zeros([total_sequence_num, dim_output], order='C')

    for k in range(mlp_input_data.shape[0] - DIM_SEQUENCE_LENGTH):
        lstm_input_data[k, :, :] = np.transpose(
            mlp_input_data[k:(k + DIM_SEQUENCE_LENGTH), :])
        lstm_output_data[k, :] = mlp_output_data[k + DIM_SEQUENCE_LENGTH, :]

    logging.info('mlp_input_data shape: {}'.format(mlp_input_data.shape))
    logging.info('mlp_output_data shape: {}'.format(mlp_output_data.shape))
    logging.info('lstm_input_data shape: {}'.format(lstm_input_data.shape))
    logging.info('lstm_output_data shape: {}'.format(lstm_output_data.shape))

    feature = [
        ("mlp_data", (mlp_input_data, mlp_output_data)),
        ("lstm_data", (lstm_input_data, lstm_output_data))
    ]
    return feature
