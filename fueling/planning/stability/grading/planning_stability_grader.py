#!/usr/bin/env python

import json
import sys
import os
from collections import defaultdict
from os import listdir
from os.path import isfile, join

from fueling.planning.stability.libs import grade_table_utils
from fueling.planning.stability.libs.planning_av import PlanningAv
from fueling.planning.stability.libs.planning_jerk import PlanningJerk
from fueling.planning.stability.libs.planning_lat_jerk import PlanningLatJerk
from fueling.planning.stability.libs.record_reader import RecordItemReader


class PlanningStabilityGrader(object):
    def __init__(self):
        self.key_lat_jerk_av = grade_table_utils.KEY_LAT_JERK_AV_SCORE
        self.key_lon_jerk_av = grade_table_utils.KEY_LON_JERK_AV_SCORE

        self.planning_lat_jerk_processor = PlanningLatJerk()
        self.planning_lon_jerk_processor = PlanningJerk()
        self.planning_av_processor = PlanningAv()
        self.score_list = []
        self.lat_jerk_av_table = defaultdict(lambda: defaultdict(int))
        self.lon_jerk_av_table = defaultdict(lambda: defaultdict(int))

        table_path = os.path.dirname(os.path.realpath(__file__))
        table_path_file = os.path.join(table_path, "reference_grade_table.json")

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

        self.lat_jerk_av_table[lat_jerk][angular_velocity] += 1
        self.lon_jerk_av_table[lon_jerk][angular_velocity] += 1

        if lat_jerk in self.grade_table[self.key_lat_jerk_av]:
            if angular_velocity in self.grade_table[self.key_lat_jerk_av][lat_jerk]:
                score_lat_jerk_av = self.grade_table[self.key_lat_jerk_av][lat_jerk][angular_velocity]
        if lon_jerk in self.grade_table[self.key_lon_jerk_av]:
            if angular_velocity in self.grade_table[self.key_lon_jerk_av][lon_jerk]:
                score_lon_jerk_av = self.grade_table[self.key_lon_jerk_av][lon_jerk][angular_velocity]
        return score_lat_jerk_av * score_lon_jerk_av

    def grade_message(self, planning_pb):
        self.planning_lat_jerk_processor.add(planning_pb)
        self.planning_lon_jerk_processor.add(planning_pb)
        self.planning_av_processor.add(planning_pb)

        av = self.planning_av_processor.get_lastest_av()
        if av is None:
            return
        lat_jerk = self.planning_lat_jerk_processor.get_lastest_lat_jerk()
        lon_jerk = self.planning_lon_jerk_processor.get_lastest_derived_jerk()
        if lat_jerk is not None and lon_jerk is not None:
            score = self.grade(lat_jerk, lon_jerk, av)
            self.score_list.append(score)

    def grade_record_file(self, pathfile):
        reader = RecordItemReader(pathfile)
        self.planning_lat_jerk_processor = PlanningLatJerk()
        self.planning_lon_jerk_processor = PlanningJerk()
        self.planning_av_processor = PlanningAv()

        self.score_list = []
        topics = ["/apollo/planning"]
        for data in reader.read(topics):
            if "planning" in data:
                planning_pb = data["planning"]
                self.grade_message(planning_pb)
        return self.score_list

    def grade_folder(self, folder):
        folder_score_list = []
        fns = sorted([f for f in listdir(folder) if isfile(join(folder, f))])
        for fn in fns:
            pathfile = folder + "/" + fn
            score_list = self.grade_record_file(pathfile)
            # print(score_list)
            folder_score_list.extend(score_list)

        return folder_score_list

    def get_score(self):
        if len(self.score_list) == 0:
            return 0
        return sum(self.score_list) / float(len(self.score_list))


if __name__ == "__main__":

    folders = sys.argv[1:]
    for i in range(len(folders)):
        folder = folders[i]
        score_list = PlanningStabilityGrader().grade_folder(folder)
        print("-----------------------------------------------")
        print("FOLDER = ", folder)
        print("score = ", sum(score_list) / float(len(score_list)))
