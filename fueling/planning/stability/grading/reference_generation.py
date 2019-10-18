from os import listdir
from os.path import isfile, join

from fueling.planning.stability.libs import grade_table_utils
from fueling.planning.stability.libs.imu_angular_velocity import ImuAngularVelocity
from fueling.planning.stability.libs.imu_speed_jerk import ImuSpeedJerk
from fueling.planning.stability.libs.record_reader import RecordItemReader


class GradeTableGenerator:
    def __init__(self):
        self.grade_table_data = dict()
        self.grade_table_data[grade_table_utils.KEY_LAT_JERK_AV_SCORE] = {}
        self.grade_table_data[grade_table_utils.KEY_LON_JERK_AV_SCORE] = {}

    def process_data(self, lat_jerk, lon_jerk, av):
        lat_jerk = grade_table_utils.get_jerk_grid(lat_jerk)
        lon_jerk = grade_table_utils.get_jerk_grid(lon_jerk)
        av = grade_table_utils.get_angular_velocity_grid(av)

        lat_jerk_av_table = self.grade_table_data[grade_table_utils.KEY_LAT_JERK_AV_SCORE]
        lon_jerk_av_table = self.grade_table_data[grade_table_utils.KEY_LON_JERK_AV_SCORE]

        if lat_jerk in lat_jerk_av_table:
            if av in lat_jerk_av_table[lat_jerk]:
                lat_jerk_av_table[lat_jerk][av] += 1
            else:
                lat_jerk_av_table[lat_jerk][av] = 1
        else:
            lat_jerk_av_table[lat_jerk] = {}
            lat_jerk_av_table[lat_jerk][av] = 1

        if lon_jerk in lon_jerk_av_table:
            if av in lon_jerk_av_table[lon_jerk]:
                lon_jerk_av_table[lon_jerk][av] += 1
            else:
                lon_jerk_av_table[lon_jerk][av] = 1
        else:
            lon_jerk_av_table[lon_jerk] = {}
            lon_jerk_av_table[lon_jerk][av] = 1

    def process_file(self, fn):
        reader = RecordItemReader(fn)
        lat_jerk_processor = ImuSpeedJerk(is_lateral=True)
        lon_jerk_processor = ImuSpeedJerk(is_lateral=False)
        av_processor = ImuAngularVelocity()

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
                    self.process_data(lat_jerk, lon_jerk, av)

    def process_folder(self, folder):
        fns = [f for f in listdir(folder) if isfile(join(folder, f))]
        fns.sort()
        for fn in fns:
            path_file = folder + "/" + fn
            self.process_file(path_file)

    def grade_table_normalization(self):
        lat_jerk_av_cnt = self.grade_table_data[grade_table_utils.KEY_LAT_JERK_AV_SCORE]
        self._normalization(lat_jerk_av_cnt)
        lon_jerk_av_cnt = self.grade_table_data[grade_table_utils.KEY_LON_JERK_AV_SCORE]
        self._normalization(lon_jerk_av_cnt)

    def _normalization(self, jerk_av_cnt):
        total = 0
        for lat_jerk, av_cnt in jerk_av_cnt.items():
            for av, cnt in av_cnt.items():
                total += cnt

        for lat_jerk, av_cnt in jerk_av_cnt.items():
            for av, cnt in av_cnt.items():
                jerk_av_cnt[lat_jerk][av] /= float(total)

    def jerk_av_cnt_to_xyz(self, jerk_type):
        lat_jerk_av_cnt = self.grade_table_data[jerk_type]
        x = []
        y = []
        z = []
        for lat_jerk, av_cnt in lat_jerk_av_cnt.items():
            for av, cnt in av_cnt.items():
                x.append(lat_jerk)
                y.append(av)
                z.append(cnt)
        return x, y, z


if __name__ == "__main__":
    import sys
    import json
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    folders = sys.argv[1:]
    # fig, ax = plt.subplots(1, 1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    generator = GradeTableGenerator()
    for i in range(len(folders)):
        generator.process_folder(folders[i])
    generator.grade_table_normalization()

    with open("grade_table.json", 'w') as f:
        data_str = json.dumps(generator.grade_table_data)
        data_str = data_str.replace("\"-0.0\"", "\"0.0\"")
        f.write(data_str)

    x, y, z = generator.jerk_av_cnt_to_xyz(grade_table_utils.KEY_LAT_JERK_AV_SCORE)
    ax.scatter3D(x, y, z, c='b', marker='.', alpha=0.4)

    ax.set_xlabel('Lat Jerk')
    ax.set_ylabel('Angular Velocity')
    plt.show()
