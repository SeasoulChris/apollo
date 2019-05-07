"""Email notifications utils."""
#!/usr/bin/env python

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib

import colored_glog as glog


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
    host = 'smtp.office365.com'
    port = 587
    from_addr = 'apollo-data-pipeline@outlook.com'

    subject = 'Apollo data pipeline job status'
    mail_passwd = os.environ.get('APOLLO_EMAIL_PASSWD')
    if not mail_passwd:
        glog.error('No credential provided to send emails.')
        return

    smtp = smtplib.SMTP()
    try:
        smtp.connect(host, port)
        smtp.starttls()
        smtp.login(from_addr, mail_passwd)
    except Exception as ex:
        glog.error('accessing email server failed with error: {}'.format(str(ex)))
        return

    to_addrs = receivers or 'apollo-data-pipeline01@baidu.com'
    message = MIMEMultipart('alternative')
    html_page = MIMEText(get_html_content(title, title_color, content), 'html')
    message.attach(html_page)
    message['From'] = from_addr
    message['To'] = ';'.join(to_addrs)
    message['Subject'] = subject
    smtp.sendmail(from_addr, to_addrs, message.as_string())
    smtp.quit()

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
