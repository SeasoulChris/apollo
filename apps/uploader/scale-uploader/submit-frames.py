#!/usr/bin/env python

"""This script generates images and submit them with frames to Scale website"""

import glob
import optparse
import os
import urllib.parse

import boto3
import cv2
import numpy as np
import requests

import fueling.common.colored_glog as glog

import filtering_rules as rules

def list_tasks(target_dir):
    """List tasks containing frames that have not been uploaded yet"""
    glog.info('target dir: {}'.format(target_dir))
    tasks = list()
    for (root, _, files) in os.walk(target_dir):
        if any(cur_file.startswith('frame-') for cur_file in files) and \
            os.path.basename(root) == 'frames':
            cur_task = os.path.dirname(root)
            if not os.path.exists(os.path.join(cur_task, 'FRAME-UPLOADED')):
                tasks.append(cur_task)
    glog.info('todo tasks: {}'.format(tasks))
    return tasks

def list_images(target_dir, task):
    """List images end files in specified task"""
    glog.info('listing images, target dir: {}, task {}'.format(target_dir, task))
    for (root, dirs, _) in os.walk(target_dir):
        cur_dir = next((cur_dir for cur_dir in dirs if os.path.basename(cur_dir) == task), None)
        if cur_dir:
            images = list(glob.glob(r'{}/*/*compressed*'.format(os.path.join(root, cur_dir))))
            glog.info('found {} image binaries in dir {}'.format(len(images), os.path.join(root, cur_dir)))
            return images

def pickup_frames(task):
    """Pick up frames based on varies of rules"""
    frames_dir = os.path.join(task, 'frames')
    frames = sorted([os.path.join(frames_dir, frame) for frame in os.listdir(frames_dir)
                    if frame.startswith('frame-') and frame.endswith('.json')])
    glog.info('original frames: {}: {}'.format(len(frames), frames))
    rules.form_chains()
    filtered_frames = rules.RulesChain.do_filter(frames)
    glog.info('filtered frames: {}: {}'.format(len(filtered_frames), filtered_frames))
    return [os.path.basename(frame) for frame in filtered_frames]

def upload_frames(task, frames, s3_client):
    """Upload frames to AWS"""
    if os.path.exists(os.path.join(task, 'FRAME-UPLOADED')):
        glog.info('frames for task {} have been uploaded'.format(task))
        return
    frames_dir = os.path.join(task, 'frames')
    bucket = 'scale-labeling'
    for frame in os.listdir(frames_dir):
        if frame in frames:
            frame_src = os.path.join(frames_dir, frame)
            frame_dst = 'frames/{}/{}'.format(os.path.basename(task), frame)
            glog.info('uploading frame from {} to {}'.format(frame_src, frame_dst))
            s3_client.upload_file(frame_src, bucket, frame_dst)
    os.mknod(os.path.join(task, 'FRAME-UPLOADED'))

def generate_and_upload_images(task, frames, s3_client):
    """Generate rgb images based on names/links, and upload to AWS afterwards"""
    if os.path.exists(os.path.join(task, 'IMAGE-UPLOADED')):
        glog.info('images for task {} have been uploaded'.format(task))
        return
    image_task = os.path.basename(os.path.dirname(task))
    streaming_image_path = os.path.join(task[:task.index('modules')+len('modules')],
                                        'streaming/images')
    images = list_images(streaming_image_path, image_task)
    image_links = os.listdir(os.path.join(task, 'images'))
    bucket = 'scale-labeling'
    for image_name in image_links:
        image_bin = next((image_bin for image_bin in images
                          if os.path.basename(image_bin) == image_name), None)
        if not image_bin:
            raise ValueError('no image binary found in {} for the link: {}'
                             .format(image_task, image_name))
        jpg_file_path = '{}/{}.jpg'.format(os.path.join(task, 'images'), image_name)
        with open(image_bin, "rb") as image_bin_file:
            data = image_bin_file.read()
            img = np.asarray(bytearray(data), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            cv2.imwrite(jpg_file_path, img)
        image_dst = 'images/{}.jpg'.format(image_name)
        glog.info('uploading picture to AWS: {}'.format(image_dst))
        s3_client.upload_file(jpg_file_path, bucket, image_dst)
    os.mknod(os.path.join(task, 'IMAGE-UPLOADED'))

def send_scale_request(task, frames):
    """Send scale requests with frames in AWS"""
    attachments = []
    frame_url = 'https://s3-us-west-1.amazonaws.com/scale-labeling/frames'
    for frame in frames:
        frame_in_aws = '{}/{}/{}'.format(frame_url, 
                                         urllib.parse.quote(os.path.basename(task), safe=''), 
                                         frame)
        attachments.append(frame_in_aws)
    payload = {
      'callback_url': 'http://www.example.com/callback',
      'instruction': "<iframe style=\"width:100%;height:800px\" src=\"https://docs.google.com/document/d/e/2PACX-1vRcJA1TH3XYO8xW4ORcCC8XKtYJwRj-U08_QtHul8E7MRBACEGuT3KifCoc7jVKVSi_iWMyeKmFX0qI/pub?embedded=true\"></iframe>",
      'attachment_type': 'json',
      'attachments': attachments,
      'labels': ['Car','Van','Pickup Truck','Truck','Bus','Cyclist','Vehicle-Other','Pedestrian','Traffic Cone','Barricade','Crosswalk Barricade','Other Object'],
      'meters_per_unit': 1
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post("https://api.scaleapi.com/v1/task/lidarannotation",
                             json=payload,
                             headers=headers,
                             auth=('test_e5d69f9f13ac4f33bba158c339112c0c', ''))
                             #auth=('live_e347bbd04f964d148f3ff59478f2239c', ''))
    glog.info('submitted task to scale, response: {}'.format(response.json()))
    return response

def record_task_frames(task, frames, scale_response):
    """Record the tasks that have been uploaded for tracking"""
    task_record_file_path = os.path.join(task, 'response.txt')
    glog.info('recording to file: {}'.format(task_record_file_path))
    with open(task_record_file_path, 'w') as task_record_file:
        task_record_file.write('{}\n'.format(task))
        task_record_file.write('{}\n'.format(len(frames)))
        task_record_file.write('{}\n'.format(frames))
        task_record_file.write('{}\n'.format(scale_response.json()))

def create_s3_client():
    aws_ak = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_sk = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not aws_ak or not aws_sk:
        raise ValueError('no AWS AK or SK provided')
    return boto3.client('s3', aws_access_key_id=aws_ak, aws_secret_access_key=aws_sk)

def main():
    """Main function"""
    parser = optparse.OptionParser()
    parser.add_option("-t", "--target",
                      help="specify the target dir, under which all tasks will be uploaded")
    (opts, _args) = parser.parse_args()
    if not opts.target:
        parser.print_help()
        return
    if opts.target.find('modules') == -1:
        glog.error('the target dir should at least reach "modules" level')
        return

    tasks = list_tasks(opts.target)
    for task in tasks:
        frames = pickup_frames(task)
        if not frames:
            glog.error('no enough frames generated for task {}'.format(task))
            continue
        # Upload frames to AWS
        s3_client = create_s3_client()
        upload_frames(task, frames, s3_client)
        # Generate images and upload to AWS
        generate_and_upload_images(task, frames, s3_client)
        # Send scale request and get response
        response = send_scale_request(task, frames)
        # Record the task id along with task name and frames list
        record_task_frames(task, frames, response)
        glog.info('successfully submitted task {}'.format(task))
    glog.info('All Done. Submitted {} tasks'.format(len(tasks)))
    
if __name__ == '__main__':
    main()
