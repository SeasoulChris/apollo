#!/usr/bin/env python

parameters = {
    'config': {
        'need_to_label': True,
        'maximum_observation_time': 8.0
    },
    'mlp': {
        'train_data_rate': 0.8,
        'size_obstacle_feature': 22,
        'size_lane_sequence_feature': 40,
        'dim_input': 22 + 40,
        'dim_hidden_1': 30,
        'dim_hidden_2': 15,
        'dim_output': 1
    },
    'cruise_mlp': {
        'dim_input': 23 + 5 * 9 + 8 + 20 * 4,
        'dim_hidden_1': 50,
        'dim_hidden_2': 18,
        'dim_output': 3
    },
    'junction_mlp': {
        'dim_input': 3 + 60,
        'dim_hidden_1': 30,
        'dim_hidden_2': 15,
        'dim_output': 12
    },
    'feature': {
        'threshold_label_time_delta': 1.0,
        'prediction_label_timeframe': 3.0,
        'maximum_maneuver_finish_time': 3.0,

        # Lane change is defined to be finished if the ratio of deviation
        # from center-line to the lane width is within this: (must be < 0.5)
        'lane_change_finish_condition': 0.1,
        'maximum_observation_time': 9.0
    },
    'offset_x': 439700,
    'offset_y': 4433250
}

labels = {'go_false': 0, 'go_true': 1, 'cutin_false': -1, 'cutin_true': 2}
