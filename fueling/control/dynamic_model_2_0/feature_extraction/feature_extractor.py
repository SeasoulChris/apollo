#!/usr/bin/python

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


class DynamicModel20FeatureExtractor():
    """ feature visualization """

    def __init__(self, file_name, secondary_file=None, origin_prefix=None, target_prefix=None):
        super().__init__()
        self.chassis_data = []
        self.pose_data = []
        self.file_name = file_name
        self.secondary_file = secondary_file
        self.origin_prefix = origin_prefix
        self.target_prefix = target_prefix
        self.chassis_idx = []
        self.pose_idx = []

    def extract_data_from_record(self):
        """ extract chassis and localization data from record"""
        self.get_chassis_data(self.file_name)
        self.get_localization_data(self.file_name)
        if self.secondary_file:
            logging.info(self.secondary_file)
            self.get_chassis_data(self.secondary_file)
            self.get_localization_data(self.secondary_file)

    def get_localization_data(self, file_name):
        """ proto to list of numpy arrays """
        localization_msgs_reader = record_utils.read_record([record_utils.LOCALIZATION_CHANNEL])
        for msg in localization_msgs_reader(file_name):
            localization = record_utils.message_to_proto(msg)
            if localization is None:
                return
            else:
                time, data = self.localization_msg_to_data(localization)
                self.pose_data.append(data)

    def get_chassis_data(self, file_name):
        """ proto to list of numpy arrays """
        chassis_msgs_reader = record_utils.read_record([record_utils.CHASSIS_CHANNEL])
        for msg in chassis_msgs_reader(file_name):
            chassis = record_utils.message_to_proto(msg)
            if chassis is None:
                return
            else:
                time, data = self.chassis_msg_to_data(chassis)
                self.chassis_data.append(data)

    def chassis_msg_to_data(self, chassis):
        """ extract numpy array from proto"""
        return (chassis.header.timestamp_sec,
                np.array([
                    chassis.header.timestamp_sec,
                    chassis.speed_mps,  # 14 speed
                    chassis.throttle_percentage / 100,  # 15 throttle
                    chassis.brake_percentage / 100,  # 16 brake
                    chassis.steering_percentage / 100,  # 17
                    chassis.driving_mode,  # 18
                    chassis.gear_location,  # 22
                ]))

    def localization_msg_to_data(self, pose):
        """ extract numpy array from proto"""
        header_time = pose.header.timestamp_sec
        pose = pose.pose
        return (header_time,
                np.array([
                    header_time,
                    pose.heading,  # 0
                    pose.orientation.qx,  # 1
                    pose.orientation.qy,  # 2
                    pose.orientation.qz,  # 3
                    pose.orientation.qw,  # 4
                    pose.linear_velocity.x,  # 5
                    pose.linear_velocity.y,  # 6
                    pose.linear_velocity.z,  # 7
                    pose.linear_acceleration.x,  # 8
                    pose.linear_acceleration.y,  # 9
                    pose.linear_acceleration.z,  # 10
                    pose.angular_velocity.x,  # 11
                    pose.angular_velocity.y,  # 12
                    pose.angular_velocity.z,  # 13
                    pose.position.x,  # 19
                    pose.position.y,  # 20
                    pose.position.z,  # 21
                ]))

    def data_pairing(self):
        """ simplified alignment"""
        if not self.chassis_data:
            return
        # chassis dt around 0.011
        # pose dt around 0.01
        # first pose msg exists when chassis at 3rd msg
        # alignment based on chassis (no interpolation for chassis)
        chassis_data = np.array(self.chassis_data)
        chassis_times = chassis_data[:, 0]
        logging.info(f'chassis_times shape is {chassis_times.shape}')
        pose_data = np.array(self.pose_data)
        pose_times = pose_data[:, 0]
        logging.info(f'pose_times shape is {pose_times.shape}')
        # for each chassis time find the nearest pose time that not chosen
        # if not matching point is found (?) a. drop the point b. interplated pose
        t_buffer = 0.011  # get from chassis frequency (dt) unit: s
        pose_time_ids = []
        chassis_id_removed = []
        first_matching_point = False
        last_matching_point = False

        for idx, chassis_time in enumerate(chassis_times):
            # search in a range
            if idx == 0:
                dt_prev = t_buffer / 2
            else:
                dt_prev = (chassis_times[idx] - chassis_times[idx - 1]) / 2
            if idx == chassis_times.shape[0] - 1:
                dt_next = t_buffer / 2
            else:
                dt_next = (chassis_times[idx + 1] - chassis_times[idx]) / 2

            pose_data_range = np.where(np.logical_and(pose_times <= (
                chassis_time + dt_next), pose_times >= (chassis_time - dt_prev)))

            if pose_data_range[0].size == 0 or pose_times[pose_times >= chassis_time].size == 0:
                # no matching pose msg for this chassis msg
                # extend searching range
                if not first_matching_point:
                    # remove first several chassis msgs
                    continue
                else:
                    # first matching point is found
                    # extend searching range
                    # TODO (might include repeats)
                    extend_range_times = 2
                    logging.debug(f'removed idx: {idx}')
                    while pose_data_range[0].size == 0:
                        pose_data_range = np.where(np.logical_and(pose_times <= (
                            chassis_time + dt_next * extend_range_times),
                            pose_times >= (chassis_time - dt_prev * extend_range_times)))
                        extend_range_times += 1
                    pose_cur_idx = np.argmin(abs(pose_times[pose_data_range] - chassis_time))
                    self.chassis_idx.append(idx)
                    self.pose_idx.append(pose_data_range[0][0])
            # more than one pose data in range chose the nearest one
            elif pose_data_range[0].size > 1:
                if not first_matching_point:
                    first_matching_point = True
                # search for nearest data point
                # not repeated
                pose_cur_idx = np.argmin(abs(pose_times[pose_data_range] - chassis_time))
                pose_time_ids.append(pose_data_range[0][pose_cur_idx])
                logging.debug(f'matching chassis {idx} is {pose_data_range[0][pose_cur_idx]}')
                self.chassis_idx.append(idx)
                self.pose_idx.append(pose_data_range[0][0])
            else:
                if not first_matching_point:
                    first_matching_point = True
                pose_time_ids.append(pose_data_range[0][0])
                logging.debug(f'matching chassis {idx} is {pose_data_range[0][0]}')
                self.chassis_idx.append(idx)
                self.pose_idx.append(pose_data_range[0][0])

    def save_as_npy(self):
        if not self.chassis_data:
            return
        features = []
        logging.info(
            f'chassis data length is {len(self.chassis_idx)}; pose data length is {len(self.pose_idx)}')
        for chassis_idx, pose_idx in zip(self.chassis_idx, self.pose_idx):
            logging.debug(f'chassis index is {chassis_idx} and pose index is {pose_idx}')
            features.append(
                self.feature_combine(self.chassis_data[chassis_idx], self.pose_data[pose_idx]))
        # write to npy
        dst_dir = os.path.dirname(self.file_name)
        file_name = f'{os.path.basename(self.file_name)}'
        dst_file = os.path.join(dst_dir, file_name + '_features.npy')
        logging.info(f'{len(features)} data points are saved to file {dst_file}')
        np.save(dst_file, np.array(features))

    def extract_features(self, data_length=100):
        """ write paired chassis and localization msgs to hdf5 files"""
        if not self.chassis_data:
            return
        chassis_idx = np.array(self.chassis_idx)
        pose_idx = np.array(self.pose_idx)
        for idx in range(0, chassis_idx.shape[0] - data_length):
            cur_feature = []
            for shift_idx in range(0, data_length):
                cur_feature.append(
                    self.feature_combine(self.chassis_data[chassis_idx[idx + shift_idx]],
                                         self.pose_data[pose_idx[idx + shift_idx]]))
            # write to hdf5 file with time stamp as tag
            file_name = "chassis{:.3f}".format(self.chassis_data[chassis_idx[idx + shift_idx]][0])
            if not self.origin_prefix or not self.target_prefix:
                file_path = os.path.join(os.path.dirname(self.file_name), 'features')
            else:
                file_path = os.path.dirname(self.file_name).replace(
                    self.origin_prefix, self.target_prefix)
            h5_utils.write_h5_single_segment(np.array(cur_feature), file_path, file_name)

    def feature_combine(self, chassis, pose):
        """ compbine chassis and localization data"""
        chassis_data = np.array(chassis)
        pose_data = np.array(pose)
        return np.hstack((pose_data[1:15, ], chassis_data[1:6, ], pose_data[15:, ],
                          chassis_data[6, ]))

    def visualizer(self, data_range_length=100, plot=False):
        """generate plots"""
        if not self.chassis_data:
            return
        chassis_data = np.array(self.chassis_data)
        pose_data = np.array(self.pose_data)
        # data length
        data_length = min(chassis_data.shape[0], pose_data.shape[0])
        chassis_time_diff = np.diff(chassis_data[:, 0])
        logging.info(f'maximum chassis delta_t is {max(chassis_time_diff)}')
        logging.info(f'minimum chassis delta_t is {min(chassis_time_diff)}')
        pose_time_diff = np.diff(pose_data[:, 0])
        logging.info(f'maximum localization delta_t is {max(pose_time_diff)}')
        logging.info(f'minimum localization delta_t is {min(pose_time_diff)}')
        if self.chassis_idx:
            chassis_idx = np.array(self.chassis_idx)
            chassis_chosen_timestamp = chassis_data[chassis_idx, 0]
            pose_idx = np.array(self.pose_idx)
            pose_chosen_timestamp = pose_data[pose_idx, 0]
            logging.info(f'chosen chassis messages total number is: {chassis_idx.shape}')
            logging.info(f'chosen localization messages total number is: {pose_idx.shape}')
            logging.info(
                f'time difference between paired chassis and localization msgs is {max(abs(chassis_chosen_timestamp-pose_chosen_timestamp))}')
            logging.info(
                f'time difference between paired chassis and localization msgs is {np.mean(abs(chassis_chosen_timestamp-pose_chosen_timestamp))}')
        if plot:
            # loop over all range
            data_range_ends = np.arange(data_range_length, data_length, data_range_length)
            for data_range_end in data_range_ends:
                cur_range = range(data_range_end - data_range_length, data_range_end)
                logging.info(f'{data_range_end} of {data_range_ends[-1]}')
                # get pose range based on chassis time range
                pose_data_range = np.where(np.logical_and(
                    pose_data[:, 0] >= chassis_data[data_range_end - data_range_length, 0],
                    pose_data[:, 0] <= chassis_data[data_range_end - 1, 0]))
                cur_pose_data = pose_data[pose_data_range]
                fig = self.plot_indexed_msgs(chassis_data[cur_range, 0],
                                             cur_pose_data[:, 0], data_range_length, data_range_end)
                # dt distribution in current range
                self.plot_time_differences(fig, 2,
                                           np.expand_dims(chassis_data[cur_range, 0], axis=-1),
                                           'Canbus msgs dt distribution', data_range_end)
                self.plot_time_differences(fig, 3,
                                           np.expand_dims(cur_pose_data[:, 0], axis=-1),
                                           'Localization msgs dt distribution', data_range_end)
                figure_file_name = f'{data_range_end}.png'
                figure_file_path = os.path.join(os.path.dirname(
                    self.file_name), 'plots', figure_file_name)
                if not os.path.exists(os.path.join(os.path.dirname(self.file_name), 'plots')):
                    os.makedirs(os.path.join(os.path.dirname(self.file_name), 'plots'))

                logging.info(f'figure saved as : {figure_file_path}')
                plt.savefig(figure_file_path)
                plt.close(fig)

    def plot_time_differences(self, fig, fig_index, data, title, index=None):
        """plot samplint time of data"""
        logging.info(data.shape)
        xdata = np.arange(data.shape[0] - 1)
        ydata = np.diff(data[:, 0])
        ax = fig.add_subplot(3, 1, fig_index)
        plt.title(title)
        plt.ylabel('Time (ms)', fontdict={'size': 12})
        plt.xlabel('Number', fontdict={'size': 12})
        plt.plot(xdata, ydata, 'b.')
        return fig

    def plot_indexed_msgs(self, chassis_timestamp, localization_timestamp, data_range_length, index=None):
        """ plot time alignment between two types of msgs"""
        fig = plt.figure(figsize=(24, 16))
        ax = fig.add_subplot(3, 1, 1)
        plt.xlabel('Time (ms)', fontdict={'size': 10})
        idx = 0
        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                               x=-0.05, y=0.10, units='inches')
        plt.plot(chassis_timestamp, np.ones((data_range_length,)),
                 'b.', label='Chassis message timestampes')
        for x, y in zip(chassis_timestamp, np.ones((data_range_length,))):
            plt.text(x, y, f'{idx}', color='b', size=6, transform=trans_offset)
            idx += 1
        idx = 0
        plt.plot(localization_timestamp, np.ones((localization_timestamp.shape[0],)),
                 'r.', label='Localization message timestampes')
        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                               x=-0.05, y=-0.20, units='inches')
        for x, y in zip(localization_timestamp, np.ones((localization_timestamp.shape[0],))):
            plt.text(x, y, f'{idx}', color='r', size=6, transform=trans_offset)
            idx += 1
        plt.legend(fontsize=10, numpoints=3, frameon=False)
        title = f'Data alignment'
        plt.title(title)
        return fig


if __name__ == '__main__':
    # each folder contained a completed trip, traverse all trip folders
    # for golden set, no more than 2 files in a folder
    file_path = 'fueling/control/dynamic_model_2_0/testdata/golden_set'
    file_list = glob.glob(os.path.join(file_path, '*/*00000.recover'))
    logging.info(f'total {len(file_list)} files: {file_list}')
    for file_name in file_list:
        feature_extractor = DynamicModel20FeatureExtractor(
            file_name, origin_prefix='golden_set', target_prefix='golden_set_features')
        feature_extractor.extract_data_from_record()
        feature_extractor.data_pairing()
        feature_extractor.save_as_npy()
        feature_extractor.extract_features()
    feature_extractor.visualizer()
