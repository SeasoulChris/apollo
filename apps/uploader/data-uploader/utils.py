#!/usr/bin/env python3

"""Provide utility functions"""

import errno
import os
import re
import subprocess

import yaml
        
def check_output(command):
    """Execute system command and returns code"""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
    output_lines = process.stdout.readlines()
    process.communicate()
    return [line.strip() for line in output_lines]

def makedirs(dir_path):
    """Make directories recursively."""
    if os.path.exists(dir_path):
        return True
    try:
        os.makedirs(dir_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            return False
    return True

def has_end_files(path):
    """If any end files under given path"""
    for end_file in os.listdir(path):
        if not os.path.isdir(os.path.join(path, end_file)):
            return True
    return False

def get_all_directories(root, excludes):
    """Recursively get all directories under given root"""
    results = []
    for path_name in os.listdir(root):
        cur_path = os.path.join(root, path_name)
        if os.path.isdir(cur_path) and not any(cur_path.find(e) != -1 for e in excludes):
            if has_end_files(cur_path):
                results.append(cur_path)
            else:
                results += get_all_directories(cur_path, excludes)
    return results

def clean_symbolic_links(path):
    """Clean symbolic links that do not exist"""
    for end_file in os.listdir(path):
        end_file = os.path.join(path, end_file)
        if os.path.islink(end_file):
            if not os.path.exists(end_file):
                os.unlink(end_file)

def get_size(path):
    """Get size of given path"""
    output = check_output('du -sh {}'.format(path))
    return output[0].split()[0].decode('utf-8')

def get_size_in_mb(size_str):
    """Get size num in MB from size string, like 24G returns 24 * 1024 MB"""
    if size_str == '0' or not size_str:
        return 0
    size_num = float(size_str[: - 1])
    size_unit = size_str[len(size_str) - 1 :]
    if size_unit.upper() == 'T':
        size_num = size_num * 1024 * 1024
    elif size_unit.upper() == 'G':
        size_num = size_num * 1024
    elif size_unit.upper() == 'K':
        size_num = size_num / 1024
    elif size_unit >= '0' and size_unit <= '9':
        size_num = float(size_str) / 1024 / 1024
    return size_num

def get_readable_time(time):
    """Convert seconds to h:m:s"""
    return '{:d}h:{:d}m:{:d}s'.format(int(time / 60 / 60),
                                      int(time /60 % 60),
                                      int(time % 60))

def get_time_seconds(time_str):
    """Convert h:m:s to seconds"""
    se = re.search(r'^(\d+)h:(\d+)m:(\d+)s$', time_str.strip(), re.M|re.I)
    return int(se.group(1)) * 60 * 60 + int(se.group(2)) * 60 + int(se.group(3))

def get_speed(time_spent, size_str):
    """Get speed based on size and spent time"""
    size_num = get_size_in_mb(size_str)
    return '{:.2f} MB/s'.format(size_num / (time_spent if time_spent != 0 else 1))

def get_spent_time(size_str, speed):
    """Get spent time based on size and speed"""
    size_num = get_size_in_mb(size_str)
    time_spent = int(size_num / speed)
    return get_readable_time(time_spent)

def load_yaml_settings(yaml_file_name):
    """Load settings from YAML config file."""
    if yaml_file_name is None:
        return None
    yaml_file = open(yaml_file_name)
    return yaml.safe_load(yaml_file)

def get_recipients(email_to_all):
    """Load emails from conf"""
    recipients = []
    if not email_to_all:
        return recipients
    yaml_settings = load_yaml_settings('conf/uploader_conf.yaml')
    if yaml_settings:
        recipients.extend([x['email'] for x in yaml_settings['recipients']])
    return recipients

def get_disk_title_object(disk_label):
    """Check disk serial number and returns its corresponding label on the disk"""
    disk_types = ('scsi', 'sat')
    for disk_type in disk_types:
        cmd = 'smartctl -i {} -d {} | grep Serial'.format(disk_label, disk_type)
        output = check_output(cmd)
        if output:
            serial = str(output[0]).split(':')[1].strip().rstrip("\'")
            yaml_settings = load_yaml_settings('conf/uploader_conf.yaml')
            if yaml_settings:
                for disk in yaml_settings['disks']:
                    if serial == disk['serial']:
                        return ' {}'.format(disk['label'])
    return ''

def find_task_path(ocean, needle):
    """DFS the needle from ocean"""
    for path in os.listdir(ocean):
        if os.path.isdir(os.path.join(ocean, path)) and path == needle:
            return os.path.join(ocean, path)
        elif os.path.isdir(os.path.join(ocean, path)):
            found = find_task_path(os.path.join(ocean, path), needle)
            if found:
                return found
    return None

def read_file_lines(file_path):
    """List all lines for a specific file"""
    with open(file_path) as read_file:
        lines = read_file.readlines()
    return [line.strip() for line in lines]
