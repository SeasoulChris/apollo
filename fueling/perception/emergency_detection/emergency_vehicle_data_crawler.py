#!/usr/bin/env python

from __future__ import unicode_literals
import os
import os.path
import subprocess as sp
import time
from datetime import datetime


import youtube_dl
from youtube_api import YouTubeDataAPI
import cv2


from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils

# Util set up
api_key = 'AIzaSyBhsHxsV1zGJdpdCVB8eDxpRADzdj6IdMg'
try:
    yt = YouTubeDataAPI(api_key)
except Exception as e:
    logging.error(e)
curr_abs_path = os.path.dirname(os.path.abspath(__file__))[37:]
bos_tool_path = '/fuel/apps/local'
bos_file_path = 'modules/perception/emergency_detection/data/emergency_vehicle'
video_log_file = 'videoURL.txt'
audio_log_file = 'audioURL.txt'
run_bos_fstool = './bos_fstool'
image_folder = 'image_clips'

# get today's date and time for URL.txt files
now = datetime.now()
dt_string = now.strftime("%m-%d-%Y_%H:%M:%S")
print(curr_abs_path)

##################################


class EmergencyVehicleDataCrawler(BasePipeline):

    def run(self):
        time_start = time.time()
        keyword = "fire truck responding"
        audio_location = "data/FireVid"
        num_of_results = 1
        self.to_rdd(range(1)).foreach(
            lambda instance: self.downloadVideo(
                keyword,
                audio_location,
                num_of_results
            )
        )
        logging.info(F'Download complete in {time.time() - time_start} seconds.')

    @staticmethod
    # Video download function
    def downloadVideo(keyword, video_storage_location, number_of_results=100):
        file_utils.makedirs('{}/{}'.format(curr_abs_path, video_storage_location))
        # get search results of seach_keywords on youtube
        try:
            result = yt.search(keyword, type="video", video_duration="short", max_results=number_of_results)
        except Exception as e:
            logging.error(e)
        # video id set for checking replication and downloading
        result_id_set = set()
        f = open(
            '{}/{}/{}_videoURL.txt'.format(
                curr_abs_path, video_storage_location, dt_string), 'w')
        for item in result:
            if "child" in item["video_title"] or "toy" in item["video_title"]:
                continue
            result_id_set.add(item["video_id"])
            f.write("https://www.youtube.com/watch?v=" + item["video_id"] + '\n')
        f.close()
        logging.info(result_id_set)
        try:
            sp.Popen(['sh', '-c', 'cd {} && {} --src={}/{}/{}_{} --dst={}/{}/{}_{}'.format(
                bos_tool_path,
                run_bos_fstool,
                curr_abs_path,
                video_storage_location,
                dt_string,
                video_log_file,
                bos_file_path,
                video_storage_location,
                dt_string,
                video_log_file
            )])
        except sp.CalledProcessError:
            print("subprocess call error!")

        # download options
        ydl_vid_opts = {
            'format': 'best',
            'ignoreerrors': 'True',
            # this location need to be changed
            'outtmpl': curr_abs_path + '/' + video_storage_location + '/%(id)s.%(ext)s'
        }
        for v_id in result_id_set:
            # video download
            with youtube_dl.YoutubeDL(ydl_vid_opts) as ydl_video:
                result = ydl_video.extract_info(v_id)
                file_path_name = ydl_video.prepare_filename(result)
                file_name = file_path_name[len(curr_abs_path):]

            # create a folder for clip images
            if not os.path.exists(
                    "{}/{}/image_clips".format(
                        curr_abs_path, video_storage_location)):

                os.mkdir("{}/{}/image_clips".format(curr_abs_path, video_storage_location))

            # Video found, start clipping and storing images, 1 image per 12 frame
            if os.path.exists(file_path_name):
                logging.info("***Found {}".format(file_path_name))
                videoCapture = cv2.VideoCapture(file_path_name)
                fps = videoCapture.get(cv2.CAP_PROP_FPS)
                success, frame = videoCapture.read()
                i = 0
                j = 0
                while success:
                    i = i + 1
                    if (i % int(fps) == 0):
                        j = j + 1
                        image_file_name = '{}_{}.jpg'.format(v_id, str(j))
                        address = '{}/{}/image_clips/{}'.format(
                            curr_abs_path,
                            video_storage_location,
                            image_file_name
                        )
                        if not cv2.imwrite(address, frame):
                            raise Exception("Could not write image")
                        else:
                            logging.info('save image: {}'.format(str(j)))
                            try:
                                sp.Popen([
                                    'sh', '-c', 'cd {} && {} --src={} --dst={}/{}/{}/{}'.format(
                                        bos_tool_path,
                                        run_bos_fstool,
                                        address,
                                        bos_file_path,
                                        video_storage_location,
                                        image_folder,
                                        image_file_name
                                    )
                                ])
                            except sp.CalledProcessError:
                                logging.info("subprocess call error!")
                    success, frame = videoCapture.read()
                try:
                    sp.Popen(['sh', '-c', 'cd {} && {} --src={}{} --dst={}{}'.format(
                        bos_tool_path,
                        run_bos_fstool,
                        curr_abs_path,
                        file_name,
                        bos_file_path,
                        file_name
                    )])
                except sp.CalledProcessError:
                    logging.error("subprocess call error!")
            else:
                logging.info("File not Found")

    @staticmethod
    # Audio download function
    def downloadAudio(keyword, audio_storage_location, number_of_results=100):
        file_utils.makedirs('{}/{}'.format(curr_abs_path, audio_storage_location))
        # audio download
        try:
            result = yt.search(keyword, type="video", max_results=number_of_results)
        except Exception as e:
            logging.error(e)
        # video id set for checking replication and downloading
        result_id_set = set()
        f = open(
            '{}/{}/{}_audioURL.txt'.format(
                curr_abs_path, audio_storage_location, dt_string), 'w')
        for item in result:
            if "child" in item["video_title"] or "toy" in item["video_title"]:
                continue
            result_id_set.add(item["video_id"])
            f.write("https://www.youtube.com/watch?v=" + item["video_id"] + '\n')
        f.close()

        try:
            sp.Popen(['sh', '-c', 'cd {} && {} --src={}/{}/{}_{} --dst={}/{}/{}_{}'.format(
                bos_tool_path,
                run_bos_fstool,
                curr_abs_path,
                audio_storage_location,
                dt_string,
                audio_log_file,
                bos_file_path,
                audio_storage_location,
                dt_string,
                audio_log_file
            )])
            # Don't remove local data file for now
            # os.remove('{}/audioURL.txt'.format(audio_storage_location))
        except sp.CalledProcessError:
            logging.error("subprocess call error!")

        print(result_id_set)

        # download options
        ydl_aud_opts = {
            'format': 'bestaudio/best',
            'ignoreerrors': 'True',
            'outtmpl': curr_abs_path + "/" + audio_storage_location + '/%(id)s.%(ext)s'
        }
        for v_id in result_id_set:
            # audio download
            with youtube_dl.YoutubeDL(ydl_aud_opts) as ydl_audio:
                result = ydl_audio.extract_info(v_id)
                file_path_name = ydl_audio.prepare_filename(result)
                file_name = file_path_name[len(curr_abs_path):]
                try:
                    sp.Popen(['sh', '-c', 'cd {} && {} --src={} --dst={}{}'.format(
                        bos_tool_path,
                        run_bos_fstool,
                        file_path_name,
                        bos_file_path,
                        file_name
                    )])
                except sp.CalledProcessError:
                    print("subprocess call error!")


if __name__ == '__main__':
    EmergencyVehicleDataCrawler().main()

# downloadVideo("police cars responding", 'data/PoliceVid', 5)
# downloadAudio("police car siren", "data/PoliceAud", 10)
# downloadVideo("fire truck responding", 'data/FireVid', 1)
# downloadAudio("fire truck siren", "data/FireAud", 10)
# downloadVideo("ambulance responding", 'data/AmbulanceVid', 1)
# downloadAudio("ambulance siren", "data/AmbulanceAud", 10)
