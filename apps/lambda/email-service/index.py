#!/usr/bin/env python

import sys

import flask
import flask_restful

from email_service import EmailService


app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(EmailService, '/')


if len(sys.argv) >= 2 and sys.argv[1] == 'debug':
    app.run(host='0.0.0.0', port=8080, debug=True)
else:
    from bae.core.wsgi import WSGIApplication
    application = WSGIApplication(app)
