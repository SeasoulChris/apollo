#!/usr/bin/env python

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import httplib
import smtplib
import sys

from flask_restful import Resource
import flask


class EmailService(Resource):
    """Email restful service"""
    PIN = "apollo2019-woyouyitouxiaomaolv"

    def post(self):
        # Request {
        #   Pin: "PIN"
        #   Title: "Title"
        #   Content: "Content"
        #   Receivers: "xiaoxiangquan@baidu.com;..."
        #   Attachments: {
        #     "Filename": "Base64 Content"
        #   }
        # }
        request = flask.request.get_json()
        # Verify Pin.
        if request.get('Pin') != PIN:
            return 'Rejected', httplib.UNAUTHORIZED
        title = request.get('Title', '')
        content = request.get('Content', '')
        receivers = request.get('Receivers')
        try:
            # TODO: Send attachments.
            self.send_outlook_email(title, content, receivers)
        except Exception as e:
            sys.stderr.write('Request={}, Exception={}\n'.format(request, e))
            return 'Failed to send email', httplib.BAD_REQUEST
        return 'Sent', httplib.OK

    @staticmethod
    def send_outlook_email(title, html_content, receivers):
        host = 'smtp.office365.com'
        port = 587
        from_addr = 'apollo-data-pipeline@outlook.com'
        password = 'ap0ll0!@#'
        subject = '[Apollo Fuel] ' + title

        smtp = smtplib.SMTP()
        try:
            smtp.connect(host, port)
            smtp.starttls()
            smtp.login(from_addr, password)
        except Exception as e:
            sys.stderr.write('accessing email server failed with error: {}\n'.format(ex))
            return
        message = MIMEMultipart('alternative')
        message.attach(MIMEText(html_content, 'html'))
        message['From'] = from_addr
        message['To'] = receivers
        message['Subject'] = subject
        smtp.sendmail(from_addr, receivers, message.as_string())
        smtp.quit()
