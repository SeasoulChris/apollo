#!/usr/bin/env python3

"""
Automatically upload files in mounted disk to specified remote server
"""

import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import os
import re
import requests
import sqlite3
import subprocess
import smtplib
import sys
import time
import traceback
import _thread
import urllib.request

def get_from_shell_raw(command, retrieve_log):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    data = None
    if retrieve_log:
        data = p.stdout.readlines()

    p.communicate()
    return data, p.returncode

def get_from_shell(command, retrieve_log=True):
    data, rc = get_from_shell_raw(command, retrieve_log)

    if data is not None:
        for i in range(len(data)):       
            data[i] = data[i].strip() # strip out white space

    return data, rc

def get_dir_size(path):
    cmd = 'sudo du -sh {p}'.format(p=path)
    data, _rc = get_from_shell(cmd)
    return data[0].split()[0].decode('utf-8')

def get_num_files(path):
    cmd = 'sudo find {p} -type f | wc -l'.format(p=path)
    data, _rc = get_from_shell(cmd)
    return int(data[0])

def dir_exists_and_non_empty(path):
    data, rc = get_from_shell('ls {p}'.format(p=path))
    if rc != 0 or len(data) == 0:
        return False
    return True

def get_recipients_from_json():
    url = "https://s3-us-west-1.amazonaws.com/scale-frames-250/recipients/recipients.json"
    recipients = []
    try:
        data = requests.get(url).json()
        for recipient in data['recipients']:
            recipients.append(recipient['email'])
    except Exception as e:
        return [], 'Failed to get recipients with error: {}'.format(str(e))
    return recipients, 'success'

def send_email_notification(result, message):
    HOST = 'email.baidu.com'
    PORT = 25
    mail_username='apollo-uploading@baidu.com'
    mail_password='' #TODO: read from env
    subject = 'Apollo data uploading status'
    from_addr = mail_username
    to_addrs, _err = get_recipients_from_json()
    if len(to_addrs) == 0:
        to_addrs = ['longtaolin@baidu.com']
    smtp = smtplib.SMTP()
    try:
        smtp.connect(HOST,PORT)
    except Exception as e:
        return False, 'Connection failed with error: {}'.format(str(e))
    smtp.starttls()
    try:
        smtp.login(mail_username,mail_password)
    except Exception as e:
        return False, 'Login failed with error: {}'.format(str(e))
    msg = MIMEMultipart('alternative')
    dict_header = ''
    for header in message[0]:
        dict_header += '<th style="text-align:center;font-family:Arial;font-size:18px;">{}</th>\n'.format(header)
    dict_header = '<tr>\n' + dict_header + '</tr>\n'
    dict_content = ''
    for row in range(1, len(message)):
        dict_content += '<tr>\n'
        for col in message[row]:
            dict_content += '<td style="text-align:left;font-family:Arial;font-size:16px;">{}</td>\n'.format(col)
        dict_content += '</tr>\n'
    colors = {'COMPLETED': "black", 'PROCESSING': "blue", 'FAILED': "red"}
    result_color = colors.get(result, 'red')
    html = '''
            <html>
            <head></head>
            <body>
            <div id="container" align="left" style="width:800px">
              <h1 align="center" style="color:%(result_color)s;">%(result)s</h1>
              <table border="1" cellspacing="0" align="center">
                <thead>
                  %(dict_header)s
                </thead>
                <tbody> 
                  %(dict_content)s
                </tbody>
              </table>
            </div>
            </body>
            </html>
        ''' % {
            'result': result,
            'result_color': result_color,
            'dict_header': dict_header,
            'dict_content': dict_content,
        }
    content = MIMEText(html, 'html')
    msg.attach(content)
    msg['From'] = from_addr
    msg['To'] = ';'.join(to_addrs)
    msg['Subject'] = subject
    smtp.sendmail(from_addr, to_addrs, msg.as_string())
    smtp.quit()
    return True

def get_size_in_mb(size):
    size_num = (float)(size[:len(size)-1])
    size_unit = size[len(size)-1:]
    if size_unit == 't' or size_unit == 'T':
        size_num = size_num * 1024 * 1024
    elif size_unit == 'g' or size_unit == 'G':
        size_num = size_num * 1024
    elif size_unit == 'k' or size_unit == 'K':
        size_num = size_num / 1024
    return size_num

def get_time_and_speed(time_spent, size):
    time_str = '{:d}h:{:d}m:{:d}s'.format((int)(time_spent/60/60), (int)(time_spent/60%60), (int)(time_spent%60))
    size_num = get_size_in_mb(size)
    speed = '{:.2f} MB/s'.format(size_num/time_spent)
    return time_str, speed

def get_estimate_time(size):
    size_num = get_size_in_mb(size)
    speed = 40.0
    time_spent = (int)(size_num/speed)
    time_str = '{:d}h:{:d}m:{:d}s'.format((int)(time_spent/60/60), (int)(time_spent/60%60), (int)(time_spent%60))
    return time_str

def find_folders(root, pattern):
  results = []
  for path_name in os.listdir(root):
    cur_path = os.path.join(root, path_name)
    if not os.path.isdir(cur_path):
      if re.match(pattern, cur_path):
        results.append(root)
        break
    else:
      results += find_folders(cur_path, pattern)
  return results

def contains_any_end_file(folder):
    for f in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, f)):
            return True
    return False
    
def get_all_directories_imp(root):  
    results = []
    for path_name in os.listdir(root):
        cur_path = os.path.join(root, path_name)
        if cur_path.endswith('UPLOADED'):
            continue
        if contains_any_end_file(cur_path):
            results.append(cur_path)
        else:
            results += get_all_directories_imp(cur_path)   
    return results

def get_all_directories(root):
    if not os.path.isdir(root) or root.endswith('UPLOADED'):
        return []
    if contains_any_end_file(root):
        return [root]
    return get_all_directories_imp(root)

def sort_by_groups(dirs, groups):
    group_map = {}
    for g in groups:
        group_map[g] = []
    rest = []
    for d in dirs:
        found = False
        for g in groups:
            if re.match(g, os.path.basename(d)):
                group_map[g].append(d)
                found = True
                break
        if not found:
            rest.append(d)
    results = []
    for g in groups:
        sub_results = sorted(group_map[g], reverse=True)
        results += sub_results
    results += sorted(rest, reverse=True)
    return results

def reorg_record_folder(path):
  file_name = os.path.basename(os.listdir(path)[0])
  YYYY = file_name[:4]
  MM = file_name[4:6]
  DD = file_name[6:8]
  hh = file_name[8:10]
  mm = file_name[10:12]
  ss = file_name[12:14]
  new_path = '{}/{}-{}-{}/{}-{}-{}-{}-{}-{}'.format(YYYY, YYYY, MM, DD, YYYY, MM, DD, hh, mm, ss)
  return new_path

def reorg_other_folder(path):
  se = re.search(r'[\S]*(\d{4})-(\d{2})-(\d{2})[\S]*', path, re.M|re.I)
  YYYY = se.group(1)
  MM = se.group(2)
  DD = se.group(3)
  new_path = '{}/{}-{}-{}/OTHER/{}'.format(YYYY, YYYY, MM, DD, os.path.basename(path))
  return new_path

def get_dest_from_src(src_folders):
  for path in src_folders:
    if (path.endswith('_s')):
      continue
    if all(re.match(r'^\d{14}.record.\d{5}$', os.path.basename(f)) for f in os.listdir(path)):
      new_path = reorg_record_folder(path)
    else:
      new_path = reorg_other_folder(path)
    yield (path, new_path)

class Logger():
    def __init__(self, file_path, notify_file_path=None):
        self._file_path = file_path
        self._notify_file_path = notify_file_path

    def _do_log(self, file_path, message):
        with open(file_path, 'a') as f:
            f.write(message + ' \n')

    def _log(self, message):
        self._do_log(self._file_path, message)
    
    def _log_notification_begin(self, message):
        self._do_log(self._notify_file_path+'-BEGIN', message)

    def _log_notification_complete(self, message):
        self._do_log(self._notify_file_path+'-COMPLETED', message)


class SqlLite3_DB():
    """
    DB operations for indicating the status of mounted disks, i.e. being processed, done processing or not started
    """
 
    def __init__(self, script_path):
        self._db_file = '{ap}/mounted_disk_db.sqlite'.format(ap=script_path)
        self._table_name = 'mounted_disks'
        self._column_id = 'id'
        self._column_path = 'path'
        self._column_status = 'status'
        self._const_initial_state = 'initial'
        self._const_processing_state = 'processing'
        self._const_done_state = 'done'
    
    def _create_table(self,logger=None):
        sql_create_table =  'CREATE TABLE IF NOT EXISTS {tn} ( \
                                        {c1} integer PRIMARY KEY AUTOINCREMENT, \
                                        {c2} text NOT NULL UNIQUE, \
                                        {c3} text NOT NULL \
                                    );'.format(tn=self._table_name, c1=self._column_id, c2=self._column_path, c3=self._column_status)       
        if logger is not None:
            logger._log('SQL- creating table: {tbl}'.format(tbl=sql_create_table))                            
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()
        c.execute(sql_create_table)
        conn.commit()
        conn.close()

    def _insert_row(self, path, logger=None):               
        if logger is not None:
            logger._log('SQL- inserting row: {v}'.format(v=path))                           
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()
        c.execute('INSERT INTO {t} ({p},{s}) VALUES ("{pv}","{sv}")'.\
            format(t=self._table_name, p=self._column_path, s=self._column_status, pv=path, sv=self._const_initial_state))
        conn.commit()
        conn.close()
        return True

    def _search_row_by_path(self, path, logger=None):
        if logger is not None:
            logger._log('SQL- searching row by path: {v}'.format(v=path))        
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()
        c.execute('SELECT * FROM {t} WHERE {p}="{pv}"'.\
            format(t=self._table_name, p=self._column_path, pv=path))
        row = c.fetchone()
        conn.commit()
        conn.close()
        return row

    def _search_all(self, logger=None):
        if logger is not None:
            logger._log('SQL- searching all')        
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()
        c.execute('SELECT * FROM {t}'.format(t=self._table_name))
        rows = c.fetchall()
        conn.commit()
        conn.close()
        return rows

    def _search_rows_by_status(self, status, logger=None):
        if logger is not None:
            logger._log('SQL- searching row by status: {v}'.format(v=status))        
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()
        c.execute('SELECT * FROM {t} WHERE {s}="{sv}"'.\
            format(t=self._table_name, s=self._column_status, sv=status))
        rows = c.fetchall()
        conn.commit()
        conn.close()
        return rows    

    def _update_row(self, path, status, logger=None):
        if logger is not None:
            logger._log('SQL- deleting row: {p}'.format(p=path))
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()

        c.execute('SELECT * FROM {t} WHERE {p}="{pv}"'.\
            format(t=self._table_name, p=self._column_path, pv=path))
        row = c.fetchone()
        if row is not None:
            c.execute('UPDATE {t} SET {s} = "{sv}" WHERE {p}="{pv}"'.\
                format(t=self._table_name, s=self._column_status, p=self._column_path, sv=status, pv=path))

        conn.commit()
        conn.close()
        return True
    
    def _delete_row(self, path, logger=None):
        if logger is not None:
            logger._log('SQL- deleting row: {p}'.format(p=path))
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()
        c.execute('DELETE FROM {t} WHERE {p}="{pv}"'.\
            format(t=self._table_name, p=self._column_path, pv=path))
        conn.commit()
        conn.close()
        return True


class Usb_device():
    def __init__(self, device_name, device_path, device_no, logger):
        self._device_name = device_name
        self._device_path = device_path
        self._device_no = device_no
        self._mount_point_prefix = '/media/apollo/apollo'
        self._mount_point = None
        self._logger = logger

    def _is_eligible_to_mount(self):
        data, _rc = get_from_shell('lsblk | grep {dn}'.format(dn=self._device_name))
        self._logger._log('Device- Is eligible to mount {d}: {r}'.format(d=self._device_name, r=data))
        if data:
            cols = data[0].split()
            if len(cols) >= 4 and cols[3].endswith(b'T'):
                return True
        return False

    def _is_mounted(self):
        data, _rc = get_from_shell('lsblk | grep {dn}'.format(dn=self._device_name))
        self._logger._log('Device- If mounted {d}: {r}'.format(d=self._device_name, r=data))
        if data:
            if len(data[0].split()) > 6:
                return True
        return False

    def _mount(self):
        self._logger._log('Device- mounting {d}'.format(d=self._device_path))
        if not self._is_mounted():
            mount_point = '{mpp}{dn}'.format(mpp=self._mount_point_prefix, dn=self._device_no)
            while dir_exists_and_non_empty(mount_point):
                self._device_no = self._device_no + 1
                mount_point = '{mpp}{dn}'.format(mpp=self._mount_point_prefix, dn=self._device_no)

            data, rc = get_from_shell('sudo mkdir -p {mp}'.format(mp=mount_point))
            if rc != 0:
                raise AssertionError('creating mount point failed for path {mp} with reason {d}'.format(mp=mount_point, d=data))
            data, rc = get_from_shell('sudo mount {dp} {mp}'.format(dp=self._device_path, mp=mount_point))
            if rc != 0:
                raise AssertionError('mount failed for path {p} with reason {d}'.format(p=self._device_path, d=data))
            self._mount_point = '{mp}'.format(mp=mount_point)
            self._logger._log('Device- mounted {d} to {p}'.format(d=self._device_path, p=self._mount_point))

    def _unmount(self):
        self._logger._log('Device- unmounting {d}'.format(d=self._device_path))
        if self._is_mounted():
            data, rc = get_from_shell('sudo umount {dp}'.format(dp=self._device_path))
            if rc != 0:
                raise AssertionError('unmount failed for path {p} with reason {d}'.format(p=self._device_path, d=data)) 
            self._mount_point = None
            self._logger._log('Device- unmounted {d}'.format(d=self._device_path)) 


class Copy_worker():
    def __init__(self, device_path, src_mount_point, logger, time_stamp):
        self._src_folder = os.path.join(src_mount_point, 'data/bag')
        self._device_folder = os.path.join(device_path, 'data/bag')
        self._dst_folder = '/mnt/bos/test'
        self._logger = logger
        self._time_stamp = time_stamp

        self._src = []
        self._dst = []

        email_message = [('Folder', 'Files', 'Size', 'Destination', 'Estimate')]
        self._logger._log('COPY- src folder {s}'.format(s=self._src_folder))

        directories = get_all_directories(self._src_folder)
        groups = (r'[\S]*_s$', r'[\S]*\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$', r'[\S]*gpsbin$')
        src_folders = sort_by_groups(directories, groups)

        src_dst_map = get_dest_from_src(src_folders)

        total_file_num = 0; total_size = 0.0 
        for src_dst_path in src_dst_map:
            src_size = get_dir_size(src_dst_path[0])
            src_file_num = get_num_files(src_dst_path[0])
            dst_path = '{}/{}'.format(self._dst_folder, src_dst_path[1])
            counter = 0
            while os.path.exists(dst_path):
                dst_path = '{}_{}'.format(dst_path, counter)
                counter += 1
            log_src_info = 'COPY- looking to copy src {s}[{ss}:{sfn}] - dst {dst}'.format(s=src_dst_path[0], ss=src_size, sfn=src_file_num, dst=dst_path)
            self._logger._log(log_src_info)
            
            self._src.append([src_dst_path[0], src_size, src_file_num])
            email_message.append((src_dst_path[0], src_file_num, src_size, dst_path, get_estimate_time(src_size)))
            total_file_num += (int)(src_file_num); total_size += get_size_in_mb(src_size)
            self._dst.append([dst_path,0,0])
        email_message.append(('Total', total_file_num, '{:.1f}G'.format(total_size/1024), '', get_estimate_time('{}M'.format(total_size))))
        send_email_notification('PROCESSING', email_message)

    def _copy_all(self, email_message):
        len_folders = len(self._src)
        self._logger._log('COPY- copying all {ln} folders'.format(ln=len_folders))
        if len_folders > 0:
            for index in range(len(self._src)):
                self._copy(email_message, index)
            self._remove_src()

    def _remove_src(self):
        move_to_dir = os.path.dirname(self._src_folder)
        for folder in os.listdir(self._src_folder):
            move_to_data_name = '{}-UPLOADED'.format(os.path.basename(folder))
            data, rc = get_from_shell('sudo mv {} {}'.format(os.path.join(self._src_folder, folder),  os.path.join(move_to_dir, move_to_data_name)))
            if rc != 0:
                self._logger._log('COPY- Failed to move src folder after copy: {r}'.format(r=data))

    def _copy(self, email_message, index=0):
        if self._src[index][2] == 0:
            self._logger._log('COPY- no files in folder {p}.  SKIP'.format(p=self._src[index][0]))
            return

        time_start = time.time()
        retries = 3

        mkdir_command = 'mkdir -p {dp}'.format(dp=self._dst[index][0])
        self._logger._log('COPY- {m}'.format(m=mkdir_command))
        data, rc = get_from_shell(mkdir_command)
        if rc != 0:
            raise AssertionError('create dst dir {dp} failed with reason {r}'.format(dp=self._dst[index][0], r=data))

        rsync_command = 'rsync -ah --append-verify --progress --no-perms --omit-dir-times {sp}/ {dp}/'.format(sp=self._src[index][0], dp=self._dst[index][0])
        self._logger._log('COPY- copying {r}'.format(r=rsync_command))
        self._logger._log_notification_begin('COPY- copying {r}'.format(r=rsync_command))
        while retries != 0:
            retries -= 1
            _data, rc = get_from_shell(rsync_command)
            #self._logger._log('COPY- retry {rt} \nrsync result: \n{rd}'.format(rt=retries, rd=_data))
            if rc == 0:
                self._logger._log('COPY- rsync completed')
                break
            elif retries == 0:
                raise AssertionError('failed to copy data from {sp} to {dp} with reason {r}'.format(sp=self._src[index][0], dp=self._dst[index][0], r=_data))
            time.sleep(3)
        
        data, rc = get_from_shell('touch {}/COMPLETE'.format(self._dst[index][0]))
        if rc != 0:
            self._logger._log('COPY- unable to mark COMPLETE: {r}'.format(r=data))

        data, _rc = get_from_shell('du -sh {dp}'.format(dp=self._dst[index][0]))
        self._dst[index][1] = data[0].split()[0].decode('utf-8')
        data, _rc = get_from_shell('find {dp} -type f | wc -l'.format(dp=self._dst[index][0]))
        self._dst[index][2] = int(data[0])
        detail_log = 'Copied from {s}[{sds}:{snf}] to {d}[{dds}:{dnf}]'.format(s=self._src[index][0],sds=self._src[index][1],snf=self._src[index][2],d=self._dst[index][0],dds=self._dst[index][1],dnf=self._dst[index][2])
        time_spent, speed = get_time_and_speed(time.time()-time_start, self._src[index][1])
        email_message.append(('{}; {}; {}'.format(self._src[index][0], self._src[index][2], self._src[index][1]), '{}; {}; {}'.format(self._dst[index][0], self._dst[index][2], self._dst[index][1]), speed, time_spent))
        if self._dst[index][1] != self._src[index][1] or self._dst[index][2] != self._src[index][2]:
            #raise AssertionError('COPY- mismatch between src and dst after copy: {r}'.format(r=detail_log))
            self._logger._log('COPY- mismatch between src and dst after copy: {r}'.format(r=detail_log))
        else:
            self._logger._log('COPY- ALL DONE: {r}'.format(r=detail_log))
            self._logger._log_notification_complete('upload completed: {r}'.format(r=detail_log))


def thread_main(script_dir, device_path): 
    notify_dir = '/home/notifyuser/Documents/Notification'
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    log_file_path = os.path.join(script_dir+os.sep+'log', '{t}.log'.format(t=time_stamp))
    notify_file_path = '{nd}/{tm}'.format(nd=notify_dir, tm=time_stamp)

    logger = Logger(log_file_path, notify_file_path)
    logger._log('Thread Main- script path {sp}. device {d}'.format(sp=logger._file_path, d=device_path))  
    logger._log_notification_begin('uploading data from {d}'.format(d=device_path))
    
    se = re.search( r'/dev/sd([a-z])(\d).*', device_path, re.M|re.I)
    dno = (int)(se.group(2))
    ss = 'abcdefghijklmnopqrstuvwxyz'
    hno = ss.find(se.group(1))
    device_name = 'sd{dn}{nb}'.format(dn=se.group(1),nb=dno)
    dno += hno*10
    db = SqlLite3_DB(script_dir)
    completed_email_message = [('Source', 'Destination', 'Speed', 'Time')]

    try:
        device = Usb_device(device_name, device_path, dno, logger)
        if not device._is_eligible_to_mount():
            logger._log('Thread Main- device not eligible to mount {d}'.format(d=device._device_path))
            device._unmount()
            db._delete_row(device_path, logger)
            exit(1)
        logger._log('Thread Main- prepare to copy {d}'.format(d=device._device_path))
        device._mount()
        if device._mount_point is not None:
            worker = Copy_worker(device_path, device._mount_point, logger, time_stamp)
            worker._copy_all(completed_email_message)
            device._unmount()
            db._delete_row(device_path, logger)
        else:
            logger._log('Thread Main- no data to copy, exiting now')
            logger._log_notification_complete('no data copied')
    except Exception as e:
        logger._log('Thread Main- Exception caught: {ex}. stack trace: {st}'.format(ex=str(e), st=traceback.format_exc()))
        logger._log_notification_complete('Exception caught: {ex}'.format(ex=str(e)))
        failed_email_message = [('Error', str(e))]
        send_email_notification('FAILED', failed_email_message)
        device._unmount()
        db._delete_row(device_path, logger)
        exit(1)
    
    logger._log_notification_complete('\n SUCCESS')
    send_email_notification('COMPLETED', completed_email_message)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print('starting main function of AutoCopy under dir {lp}'.format(lp=script_dir))

    db = SqlLite3_DB(script_dir)
    db._create_table()
    heart_beat = 0

    while True:
        for row in db._search_rows_by_status('initial'):
            if row[2] == db._const_initial_state and re.match(r'/dev/sd[a-z]\d+.*',row[1]):
                try:
                    print ('starting copy worker thread')
                    db._update_row(row[1], db._const_processing_state)
                    _thread.start_new_thread(thread_main, (script_dir, row[1]))
                    time.sleep(1)
                except:
                    print ('Error: Cannot start thread. path={p}'.format(row[1]))
        
        if heart_beat == 0:
            print ('sleeping 30 sec...')

        heart_beat = (heart_beat+1)%10
        time.sleep(30)

    
    
    
