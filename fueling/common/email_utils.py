"""Email notifications utils."""
#!/usr/bin/env python

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib

import colored_glog as glog
import requests


BAE_PROXY = 'http://192.168.1.31'
BAE_PROXY_PIN = 'apollo2019-woyouyitouxiaomaolv'


def send_email_info(title, content, receivers=None):
    """Send email with normal information"""
    send_email(title, 'blue', content, receivers)

def send_email_warn(title, content, receivers=None):
    """Send email with warning information"""
    send_email(title, 'yellow', content, receivers)

def send_email_error(title, content, receivers=None):
    """Send email with error information"""
    send_email(title, 'red', content, receivers)

def send_email(title, title_color, content, receivers=None):
    """
    Send emails in the format of HTML for notification of job status, statistic and etc
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
       (
           'Succeeded': 100,
           'Failed': 2000,
           'Total': 2100
       )
    4. receivers, recepients of the notification. Default should be a group account, but can also
       be specified explicitly.
    """
    html_content = get_html_content(title, title_color, content)
    receivers = ';'.join(receivers)
    request_json = {
        'Pin': BAE_PROXY_PIN,
        'Title': title,
        'Content': html_content,
        'Receivers': receivers,
    }
    request = requests.post(BAE_PROXY, json=request_json)
    if request.ok:
        glog.info('Successfully send email to {}'.format(receivers))
    else:
        glog.error('Failed to send email to {}, status={}, content={}'.format(
            receivers, request.status_code, request.content))

def get_html_content(title, title_color, content):
    """Help function to constuct HTML message body"""
    header_row_prefix = '<thead>\n<tr>\n'
    header_row_suffix = '</tr>\n</thead>\n'
    header_col_prefix = '<th style="text-align:center;font-family:Arial;font-size:18px;">'
    header_col_suffix = '</th>\n'
    row_prefix = '<tr>\n'
    row_suffix = '</tr>\n'
    col_prefix = '<td style="text-align:left;font-family:Arial;font-size:16px;">'
    col_suffix = '</td>\n'
    header = None
    html_content = ''
    if not content:
        return html_content
    if isinstance(content, dict):
        rows = content.items()
    else:
        rows = [list(named_tuple) for named_tuple in content]
        header = ['{}{}{}'.format(header_col_prefix, named_tuple, header_col_suffix)
                  for named_tuple in content[0]._fields]
        header = '{}{}{}'.format(header_row_prefix, '\n'.join(header), header_row_suffix)
    for row in rows:
        html_content += row_prefix
        for col in row:
            html_content += '{}{}{}'.format(col_prefix, col, col_suffix)
        html_content += row_suffix
    # TODO: we should consider loading template files from storage if there are more than one
    # in the future.  And a better idea might be loading html directly from web application
    # frontend so we dont have to put together all the pieces here
    return '''
            <html>
            <head></head>
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
