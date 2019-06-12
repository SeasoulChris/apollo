#!/usr/bin/env python

from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import binascii
import email.encoders
import httplib
import mimetypes
import os
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
        if request.get('Pin') != self.PIN:
            return 'Rejected', httplib.UNAUTHORIZED
        subject = '[Apollo Fuel] ' + request.get('Title', '')
        content = request.get('Content', '')
        receivers = request.get('Receivers')
        attachments = {filename: binascii.a2b_base64(base64_content)
                       for filename, base64_content in request.get('Attachments', {}).items()}
        try:
            sys.stderr.write('Request={}\n'.format(request))
            self.send_outlook_email(subject, content, receivers, attachments)
        except Exception as e:
            sys.stderr.write('Request={}, Exception={}\n'.format(request, e))
            return 'Failed to send email', httplib.BAD_REQUEST
        return 'Sent', httplib.OK

    @staticmethod
    def send_outlook_email(subject, content, receivers, attachments={}):
        """Send email"""
        host = 'smtp.office365.com'
        port = 587
        from_addr = 'apollo-data-pipeline@outlook.com'
        password = 'ap0ll0!@#'
        EmailService.send_smtp_email(host, port, from_addr, password,
                                     subject, content, receivers, attachments)

    @staticmethod
    def send_smtp_email(smtp_host, smtp_port, from_addr, password,
                        subject, content, receivers, attachments={}):
        """Send email via SMTP server."""
        smtp = smtplib.SMTP()
        try:
            smtp.connect(smtp_host, smtp_port)
            smtp.starttls()
            smtp.login(from_addr, password)
        except Exception as e:
            sys.stderr.write('Accessing email server failed with error: {}\n'.format(e))
            return
        message = MIMEMultipart('alternative')
        message.attach(MIMEText(content, 'html'))
        message['Subject'] = subject
        message['From'] = from_addr
        message['To'] = receivers
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
        smtp.sendmail(from_addr, receivers.split(';'), message.as_string())
        smtp.quit()
