#!/usr/bin/env python3
"""
Modules that hold global variables, eg app ..
"""

import datetime
import flask


# Instantiate get flask APP and configuring the development environment
app = flask.Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object("conf.develop_setting.DevelopConfig")
app.secret_key = str(datetime.datetime.now())
