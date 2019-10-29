#!/usr/bin/env python

import json
import sys
import os
from os import listdir
from os.path import isfile, join

from fueling.planning.stability.libs import grade_table_utils
from fueling.planning.stability.libs.imu_angular_velocity import ImuAngularVelocity
from fueling.planning.stability.libs.imu_speed_jerk import ImuSpeedJerk
from fueling.planning.stability.libs.record_reader import RecordItemReader


class Grader:
    def __init__(self):
        self.key_lat_jerk_av = grade_table_utils.KEY_LAT_JERK_AV_SCORE
        self.key_lon_jerk_av = grade_table_utils.KEY_LON_JERK_AV_SCORE

        table_path = os.path.dirname(os.path.realpath(__file__))
        table_path_file = table_path + "/" + "grade_table.json"

        with open(table_path_file, 'r') as f:
            self.grade_table = json.loads(f.read())

    def grade(self, lat_jerk, lon_jerk, angular_velocity):
        score_lat_jerk_av = 0
        score_lon_jerk_av = 0

        angular_velocity = str(grade_table_utils.get_angular_velocity_grid(angular_velocity))
        if angular_velocity == "-0.0":
            angular_velocity = "0.0"

        lat_jerk = str(grade_table_utils.get_jerk_grid(lat_jerk))
        if lat_jerk == "-0.0":
            lat_jerk = "0.0"

        lon_jerk = str(grade_table_utils.get_jerk_grid(lon_jerk))
        if lon_jerk == "-0.0":
            lon_jerk = "0.0"

        if lat_jerk in self.grade_table[self.key_lat_jerk_av]:
            if angular_velocity in self.grade_table[self.key_lat_jerk_av][lat_jerk]:
                score_lat_jerk_av = self.grade_table[self.key_lat_jerk_av][lat_jerk][angular_velocity]
        if lon_jerk in self.grade_table[self.key_lon_jerk_av]:
            if angular_velocity in self.grade_table[self.key_lon_jerk_av][lon_jerk]:
                score_lon_jerk_av = self.grade_table[self.key_lon_jerk_av][lon_jerk][angular_velocity]
        return score_lat_jerk_av * score_lon_jerk_av

    def grade_record_file(self, folder, fn):
        reader = RecordItemReader(folder + "/" + fn)
        lat_jerk_processor = ImuSpeedJerk(is_lateral=True)
        lon_jerk_processor = ImuSpeedJerk(is_lateral=False)
        av_processor = ImuAngularVelocity()

        score_list = []
        topics = ["/apollo/localization/pose"]
        for data in reader.read(topics):
            if "pose" in data:
                pose_data = data["pose"]
                av_processor.add(pose_data)
                lat_jerk_processor.add(pose_data)
                lon_jerk_processor.add(pose_data)

                av = av_processor.get_latest_corrected_angular_velocity()
                if av is None:
                    continue
                lat_jerk = lat_jerk_processor.get_lastest_jerk()
                lon_jerk = lon_jerk_processor.get_lastest_jerk()
                if lat_jerk is not None and lon_jerk is not None:
                    score = self.grade(lat_jerk, lon_jerk, av)
                    score_list.append(score)
        return score_list

    def grade_folder(self, folder):
        folder_score_list = []
        fns = sorted([f for f in listdir(folder) if isfile(join(folder, f))])
        for fn in fns:
            score_list = self.grade_record_file(folder, fn)
            # print(score_list)
            folder_score_list.extend(score_list)

        return folder_score_list


if __name__ == "__main__":

    folders = sys.argv[1:]
    for i in range(len(folders)):
        folder = folders[i]
        score_list = Grader().grade_folder(folder)
        print("-----------------------------------------------")
        print("FOLDER = ", folder)
        print("score = ", sum(score_list) / float(len(score_list)))
