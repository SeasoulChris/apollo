#!/usr/bin/env python
"""Email notifications utils."""

import binascii
import os

import colored_glog as glog
import requests


BAE_PROXY = 'http://192.168.1.31'
BAE_PROXY_PIN = 'apollo2019-woyouyitouxiaomaolv'
DATA_TEAM = [
    'longtaolin@baidu.com',
    'xiaoxiangquan@baidu.com',
]
CONTROL_TEAM = [
    'jiaxuanxu@baidu.com',
    'luoqi06@baidu.com',
    'shujiang@baidu.com',
    'yuwang01@baidu.com',
    'runxinhe@baidu.com',
    'jinyunzhou@baidu.com',
]
PREDICTION_TEAM = [
    'hongyisun@baidu.com',
    'jiachengpan@baidu.com',
    'xukecheng@baidu.com',
]


def send_email_info(title, content, receivers, attachments=[]):
    """Send email with normal information"""
    send_email(title, 'blue', content, receivers, attachments)


def send_email_warn(title, content, receivers, attachments=[]):
    """Send email with warning information"""
    send_email(title, 'yellow', content, receivers, attachments)


def send_email_error(title, content, receivers, attachments=[]):
    """Send email with error information"""
    send_email(title, 'red', content, receivers, attachments)


def send_email(title, title_color, content, receivers, attachments=[]):
    """
    Send email in the format of HTML for notification of job status, statistic and etc
    Parameters:
    1. title, for example 'Control Feature Extraction Job Processing'
    2. title_color, got from the interface functions, like send_email_info etc
    3. content, forms the body of email, in the format of HTML tables.  Supports two formats:
       1). a list of named tuples that, for example:
       (
           (task='2019-01-05-10-10-10', target='generated', files_number=123, total_size='100G')
           (task='2019-02-22-15-15-15', target='generated', files_number=456, total_size='200G')
           ...
       )
       2). a dictionary with keys and values, for example:
       {
           'Succeeded': 100,
           'Failed': 2000,
           'Total': 2100
       }
       3). a string.
    4. receivers, recepients of the notification. Default should be a group account, but can also
       be specified explicitly.
    5. attachments, attachment files list.
    """
    html_content = get_html_content(title, title_color, content)
    receivers = ';'.join(receivers)

    base64_attachments = {}
    attachment_size = 0
    for attachment in attachments:
        try:
            with open(attachment, 'rb') as fin:
                base64_content = binascii.b2a_base64(fin.read())
                base64_attachments[os.path.basename(attachment)] = base64_content
                attachment_size += len(base64_content)
        except Exception as err:
            glog.error('Failed to add attachment {}: {}'.format(attachment, err))
    glog.info('Attached {} files with {} bytes of base64 content from {}'.format(
        len(base64_attachments), attachment_size, attachments))

    request_json = {
        'Pin': BAE_PROXY_PIN,
        'Title': title,
        'Content': html_content,
        'Receivers': receivers,
        'Attachments': base64_attachments,
    }
    request = requests.post(BAE_PROXY, json=request_json)
    if request.ok:
        glog.info('Successfully send email to {}'.format(receivers))
    else:
        glog.error('Failed to send email to {}, status={}, content={}'.format(
            receivers, request.status_code, request.content))


def get_html_content(title, title_color, content):
    """Help function to constuct HTML message body"""
    if not content:
        return ''
    if isinstance(content, str):
        return '<html><body><pre>%s</pre></body></html>' % content

    header_row_prefix = '<thead>\n<tr>\n'
    header_row_suffix = '</tr>\n</thead>\n'
    header_col_prefix = '<th style="text-align:center;font-family:Arial;font-size:18px;">'
    header_col_suffix = '</th>\n'
    header = None
    if isinstance(content, dict):
        rows = content.items()
    else:
        rows = [list(named_tuple) for named_tuple in content]
        header = ['{}{}{}'.format(header_col_prefix, named_tuple, header_col_suffix)
                  for named_tuple in content[0]._fields]
        header = '{}{}{}'.format(header_row_prefix, '\n'.join(header), header_row_suffix)

    row_prefix = '<tr>\n'
    row_suffix = '</tr>\n'
    col_prefix = '<td style="text-align:left;font-family:Arial;font-size:16px;">'
    col_suffix = '</td>\n'
    html_content = ''
    for row in rows:
        html_content += row_prefix
        for col in row:
            html_content += '{}{}{}'.format(col_prefix, col, col_suffix)
        html_content += row_suffix
    return '''
            <html>
            <body>
            <div id="container" align="left" style="width:800px">
              <h1 align="center" style="color:%(title_color)s;">%(title)s</h1>
              <table border="1" cellspacing="0" align="center">
                  %(header)s
                <tbody>
                  %(html_content)s
                </tbody>
              </table>
            </div>
            </body>
            </html>
        ''' % {
        'title_color': title_color,
        'title': title,
        'header': header or '',
        'html_content': html_content
    }
