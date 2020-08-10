#!/usr/bin/env python

SHIFT = False

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
    }
}

semantic_map_config = {
    'offset_x': 439700,
    'offset_y': 4433250,
    'map_coords': {
        'san_mateo': {
            'lower_left_x': 558980,
            'lower_left_y': 4156780,
            'horizontal_pixel_size': 11000,
            'vertical_pixel_size': 14000,
            'resolution': 0.1,
            'distance_buffer': 100
        },
        'sunnyvale': {
            'lower_left_x': 585870,
            'lower_left_y': 4139900,
            'horizontal_pixel_size': 28000,
            'vertical_pixel_size': 20000,
            'resolution': 0.1,
            'distance_buffer': 100
        },
        'sunnyvale_with_two_offices': {
            'lower_left_x': 585870,
            'lower_left_y': 4139900,
            'horizontal_pixel_size': 28000,
            'vertical_pixel_size': 20000,
            'resolution': 0.1,
            'distance_buffer': 100
        },
        'houston': {
            'lower_left_x': 244400,
            'lower_left_y': 3285100,
            'horizontal_pixel_size': 72600,
            'vertical_pixel_size': 24300,
            'resolution': 0.1,
            'distance_buffer': 200
        },
        'baidudasha': {
            'lower_left_x': 439600,
            'lower_left_y': 4433150,
            'horizontal_pixel_size': 6100,
            'vertical_pixel_size': 7100,
            'resolution': 0.1,
            'distance_buffer': 200
        },
        'XiongAn': {
            'lower_left_x': 405000,
            'lower_left_y': 4322200,
            'horizontal_pixel_size': 8700,
            'vertical_pixel_size': 10300,
            'resolution': 0.1,
            'distance_buffer': 200
        },
        'XiaMen': {
            'lower_left_x': 597600,
            'lower_left_y': 2719900,
            'horizontal_pixel_size': 11800,
            'vertical_pixel_size': 12300,
            'resolution': 0.1,
            'distance_buffer': 200
        }
    },
}

labels = {'go_false': 0, 'go_true': 1, 'cutin_false': -1, 'cutin_true': 2}

if not SHIFT:
    semantic_map_config['offset_x'] = 0
    semantic_map_config['offset_y'] = 0
