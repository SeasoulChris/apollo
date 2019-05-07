#!/usr/bin/env python

"""This script generates images and submit them with frames to Scale website"""

import glob
import json
import optparse
import os
import urllib.parse

import boto3
import colored_glog as glog
import cv2
import numpy as np
import requests
import scaleapi

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

def list_uploaded_tasks(target_dir, qualified_tasks_path):
    """List tasks that have been uploaded already"""
    glog.info('target dir: {}'.format(target_dir))
    tasks = list()
    qualified_tasks = list()
    with open(qualified_tasks_path, 'r') as qualified_tasks_file:
        qualified_tasks = [task.strip() for task in qualified_tasks_file.readlines()]
    for (root, _, files) in os.walk(target_dir):
        if any(cur_file.startswith('frame-') for cur_file in files) and \
            os.path.basename(root) == 'frames':
            cur_task = os.path.dirname(root)
            if os.path.exists(os.path.join(cur_task, 'FRAME-UPLOADED')) and \
                urllib.parse.quote(os.path.basename(cur_task), safe='') in qualified_tasks:
                tasks.append(cur_task)
    glog.info('todo tasks: {}'.format(tasks))
    return tasks

def list_images(target_dir, task):
    """List images end files in specified task"""
    images = list()
    glog.info('listing images, target dir: {}, task {}'.format(target_dir, task))
    for (root, dirs, _) in os.walk(target_dir):
        cur_dir = next((cur_dir for cur_dir in dirs if os.path.basename(cur_dir) == task), None)
        if cur_dir:
            images.extend(list(glob.glob(r'{}/*/*compressed*'.format(
                os.path.join(root, cur_dir)))))
            glog.info('{} image bins in dir {}'.format(len(images), os.path.join(root, cur_dir)))
    return images

def pickup_frames(task):
    """Pick up frames based on varies of rules"""
    frames_dir = os.path.join(task, 'frames')
    frames = sorted([os.path.join(frames_dir, frame) for frame in os.listdir(frames_dir)
                     if frame.startswith('frame-') and frame.endswith('.json')])
    glog.info('original frames: {}: {}'.format(len(frames), frames))
    filtered_frames = rules.RulesChain.do_filter(frames)
    glog.info('filtered frames: {}: {}'.format(len(filtered_frames), filtered_frames))
    return [os.path.basename(frame) for frame in filtered_frames]

def upload_frames(task, frames, s3_client):
    """Upload frames to AWS"""
    if os.path.exists(os.path.join(task, 'FRAME-UPLOADED')):
        glog.info('frames for task {} have been uploaded'.format(task))
        return
    frames_dir = os.path.join(task, 'frames')
    for frame in os.listdir(frames_dir):
        if frame in frames:
            frame_src = os.path.join(frames_dir, frame)
            frame_dst = 'frames/{}/{}'.format(os.path.basename(task), frame)
            glog.info('uploading frame from {} to {}'.format(frame_src, frame_dst))
            s3_upload_file(s3_client, frame_src, frame_dst)
    os.mknod(os.path.join(task, 'FRAME-UPLOADED'))

def generate_and_upload_images(task, s3_client):
    """Generate rgb images based on names/links, and upload to AWS afterwards"""
    if os.path.exists(os.path.join(task, 'IMAGE-UPLOADED')):
        glog.info('images for task {} have been uploaded'.format(task))
        return
    image_task = os.path.basename(os.path.dirname(task))
    streaming_image_path = os.path.join(task[:task.index('modules')+len('modules')],
                                        'streaming/images')
    images = list_images(streaming_image_path, image_task)
    image_links = os.listdir(os.path.join(task, 'images'))
    for image_name in image_links:
        if image_name.endswith('.jpg'):
            # Do not generate again if already existed
            continue
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
        s3_upload_file(s3_client, jpg_file_path, image_dst)
    os.mknod(os.path.join(task, 'IMAGE-UPLOADED'))

def get_uploaded_front6mm_images(task, frames):
    """Get uploaded images for specified task"""
    images = list()
    for frame in frames:
        with open(os.path.join(task, 'frames', frame)) as json_file:
            json_data = json.load(json_file)
            images.extend([image['image_url'] for image in json_data['images']
                           if image['image_url'].find('front_6mm') > 0])
    return images

def send_lidar_request(task, frames, access_key):
    """Send scale requests with frames in AWS"""
    attachments = []
    frame_url = 'https://s3-us-west-1.amazonaws.com/scale-labeling/frames'
    for frame in frames:
        frame_in_aws = '{}/{}/{}'.format(frame_url,
                                         urllib.parse.quote(os.path.basename(task), safe=''),
                                         frame)
        attachments.append(frame_in_aws)
    payload = {
        'project': 'scale_labeling_2019Q2',
        'callback_url': 'http://www.example.com/callback',
        'instruction': '<iframe style="width:100%;height:800px"' \
                        'src="https://docs.google.com/document/d/e/2PACX-1vRcJA1TH3XYO8xW4ORcCC8' \
                        'XKtYJwRj-U08_QtHul8E7MRBACEGuT3KifCoc7jVKVSi_iWMyeKmFX0qI/pub?' \
                        'embedded=true"></iframe>',
        'attachment_type': 'json',
        'attachments': attachments,
        'labels': ['Car', 'Van', 'Pickup Truck', 'Truck', 'Bus', 'Cyclist',
                   'Vehicle-Other', 'Pedestrian', 'Traffic Cone', 'Barricade',
                   'Crosswalk Barricade', 'Other Object'],
        'meters_per_unit': 1
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post('https://api.scaleapi.com/v1/task/lidarannotation',
                             json=payload,
                             headers=headers,
                             auth=(access_key, ''))
    glog.info('submitted task to scale, response: {}'.format(response.json()))
    return response.json()

def send_laneline_request(lidar_task_id, image_url, access_key):
    """Send scale request for LaneLine labeling"""
    client = scaleapi.ScaleClient(access_key)
    response = client.create_lineannotation_task(
        project='scale_labeling_2019Q2_laneline',
        callback_url='http://www.example.com/callback',
        instruction='<iframe src="https://docs.google.com/document/d/e/2PACX-1vQvKZEaYMG4fBKP' \
                    'JLqleaJHH26smz0TN60ur7OdIgddXlUtbmkkVNuUuopHionuapEl8eeGfp6_r5V3/pub?' \
                    'embedded=true"></iframe>',
        attachment_type='image',
        attachment=image_url,
        objects_to_annotate=['solid line', 'dashed line'],
        with_labels=True,
        splines=True,
        metadata={'lidar_task': os.path.basename(lidar_task_id)},
        annotation_attributes={
            'category': {
                'type': 'category',
                'description': 'Choose the line type:',
                'choices': [
                    'single solid',
                    'single dash',
                    'double solid',
                    'double dash',
                    'left dash right solid',
                    'left solid right dash',
                    'curb',
                    'parking lane',
                    'imaginary lane',
                    'construction cone line',
                    'other'
                ]
            },
            'ego position': {
                'type': 'category',
                'description': 'What is the position relative to the ego car?',
                'choices': [
                    '-4 left',
                    '-3 left',
                    '-2 left',
                    '-1 left',
                    '0 change lane case',
                    '1 right',
                    '2 right',
                    '3 right',
                    '4 right',
                    'other'
                ]
            },
            'fork and merge': {
                'type': 'category',
                'description': 'Does the line include a fork or merge?',
                'choices': [
                    'lane fork left',
                    'lane fork right',
                    'left lane before merge',
                    'right lane before merge',
                    'left end lane',
                    'right end lane',
                    'center lane',
                    'none'
                ]
            },
            'color': {
                'type': 'category',
                'description': 'Choose the color:',
                'choices': [
                    'white',
                    'yellow'
                ]
            }})
    response_json = json.dumps(repr(response))
    glog.info('submitted LaneLine task to scale, response: {}'.format(response_json))
    return response_json

def record_task_response(task, contents, scale_response):
    """Record the tasks that have been uploaded for tracking"""
    task_record_file_path = os.path.join(task, 'response.txt')
    glog.info('recording to file: {}'.format(task_record_file_path))
    with open(task_record_file_path, 'a') as task_record_file:
        task_record_file.write('{}\n'.format(task))
        task_record_file.write('{}\n'.format(len(contents)))
        task_record_file.write('{}\n'.format(contents))
        task_record_file.write('{}\n'.format(scale_response))

def create_s3_client():
    """Create AWS client"""
    aws_ak = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_sk = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not aws_ak or not aws_sk:
        raise ValueError('no AWS AK or SK provided')
    return boto3.client('s3', aws_access_key_id=aws_ak, aws_secret_access_key=aws_sk)

def s3_upload_file(s3_client, src, dst):
    """Upload a file to AWS"""
    bucket = 'scale-labeling'
    s3_client.upload_file(src, bucket, dst, ExtraArgs={'ACL':'public-read'})

def main():
    """Main function"""
    parser = optparse.OptionParser()
    parser.add_option("-t", "--target",
                      help="specify the target dir, under which all tasks will be uploaded")
    parser.add_option("-r", "--repeat",
                      help="specify the txt file containing tasks that we need to submit again")
    (opts, _args) = parser.parse_args()
    if not opts.target:
        parser.print_help()
        return
    if opts.target.find('modules') == -1:
        glog.error('the target dir should at least reach "modules" level')
        return
    scale_access_key_test = os.environ.get('SCALE_ACCESS_KEY_TEST')
    scale_access_key_live = os.environ.get('SCALE_ACCESS_KEY_LIVE')
    if not scale_access_key_test or not scale_access_key_live:
        raise ValueError('no scale access keys provided')
    scale_access_key = scale_access_key_live
    if opts.repeat:
        tasks = list_uploaded_tasks(opts.target, opts.repeat)
    else:
        tasks = list_tasks(opts.target)
    rules.form_chains()
    sucessful_tasks_counter = 0
    for task in tasks:
        frames = pickup_frames(task)
        if not frames:
            glog.error('no enough frames generated for task {}'.format(task))
            continue
        if not opts.repeat:
            # Upload frames to AWS
            s3_client = create_s3_client()
            upload_frames(task, frames, s3_client)
            # Generate images and upload to AWS
            generate_and_upload_images(task, s3_client)
            scale_access_key = scale_access_key_test
        # Send scale requests and record responses
        lidar_response = send_lidar_request(task, frames, scale_access_key)
        record_task_response(task, frames, lidar_response)
        front6mm_uploaded_images = get_uploaded_front6mm_images(task, frames)
        glog.info('got {} pictures from frames: {}'
                  .format(len(front6mm_uploaded_images), front6mm_uploaded_images))
        for front6mm_image in front6mm_uploaded_images:
            laneline_response = send_laneline_request(task, front6mm_image, scale_access_key)
            record_task_response(task, front6mm_uploaded_images[0], laneline_response)
        sucessful_tasks_counter += 1
        glog.info('successfully submitted task {}'.format(task))
    glog.info('All Done. Submitted {}/{} tasks'.format(sucessful_tasks_counter, len(tasks)))

if __name__ == '__main__':
    main()
