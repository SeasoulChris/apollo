#!/usr/bin/env python
"""Email notifications utils."""

from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import email.encoders
import mimetypes
import os
import smtplib
import sys

import fueling.common.logging as logging


DATA_TEAM = [
    'longtaolin@baidu.com',
    'xiaoxiangquan@baidu.com',
    'fengzongbao@baidu.com',
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

SIMPlEHDMAP_TEAM = [
    'hukuang@baidu.com',
    'v_panxuechao@baidu.com',
    'v_wangxitong02@baidu.com',
]


def send_email_info(title, content, receivers, attachments=[]):
    """Send email with normal information"""
    subject = '[Apollo Fuel] ' + title
    html_content = EmailService.get_html_content(title, 'blue', content)
    EmailService.send_email(subject, html_content, receivers, attachments)


def send_email_error(title, content, receivers, attachments=[]):
    """Send email with error information"""
    subject = '[Apollo Fuel] Error: ' + title
    html_content = EmailService.get_html_content(title, 'red', content)
    EmailService.send_email(subject, html_content, receivers, attachments)


class EmailService(object):
    """Email service"""

    @staticmethod
    def get_html_content(title, title_color, content):
        """
        Supports formats:
        1. a string.

        2. a list of named tuples, for example:
           ((task='2019-01-05-10-10-10', target='generated', files_number=123, total_size='100G')
            (task='2019-02-22-15-15-15', target='generated', files_number=456, total_size='200G')
            ...)

        3. a dictionary with keys and values, for example:
           {'Succeeded': 100,
            'Failed': 2000,
            'Total': 2100}
        """
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
                    <tbody> %(html_content)s </tbody>
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

    @classmethod
    def send_email(cls, subject, content, receivers, attachments):
        attachments_dict = {}
        for attachment in attachments:
            try:
                with open(attachment, 'rb') as fin:
                    attachments_dict[os.path.basename(attachment)] = fin.read()
            except Exception as err:
                logging.error('Failed to add attachment {}: {}'.format(attachment, err))
        cls.send_outlook_email(subject, content, receivers, attachments_dict)

    @classmethod
    def send_outlook_email(cls, subject, content, receivers, attachments={}):
        """Send email"""
        host = 'smtp.office365.com'
        port = 587
        from_addr = os.environ.get('OUTLOOK_USER')
        password = os.environ.get('OUTLOOK_PASSWD')
        cls.send_smtp_email(host, port, from_addr, password, subject, content,
                            receivers, attachments)

    @staticmethod
    def send_smtp_email(host, port, from_addr, password, subject, content,
                        receivers, attachments={}):
        """Send email via SMTP server."""
        smtp = smtplib.SMTP()
        try:
            smtp.connect(host, port)
            smtp.starttls()
            smtp.login(from_addr, password)
        except Exception as e:
            sys.stderr.write('Accessing email server failed with error: {}\n'.format(e))
            return
        message = MIMEMultipart('alternative')
        message.attach(MIMEText(content, 'html'))
        message['Subject'] = subject
        message['From'] = from_addr
        message['To'] = ';'.join(receivers)
        for filename, file_content in attachments.items():
            attachment = None
            # Create attachment with proper MIME type.
            ctype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            if maintype == 'text':
                attachment = MIMEText(file_content, _subtype=subtype)
            elif maintype == 'image':
                attachment = MIMEImage(file_content, _subtype=subtype)
            elif maintype == 'audio':
                attachment = MIMEAudio(file_content, _subtype=subtype)
            else:
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(file_content)
            email.encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
            message.attach(attachment)
        smtp.sendmail(from_addr, receivers, message.as_string())
        smtp.quit()
