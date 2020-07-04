#!/usr/bin/env python3
"""
Modules that hold global variables, eg app ..
"""

import flask

from common import filter


# Instantiate get flask APP and configuring the development environment
app = flask.Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object("conf.develop_setting.DevelopConfig")

# Register filter
app.add_template_filter(filter.get_show_job_type, "show_type")
app.add_template_filter(filter.get_action, "show_action")
app.add_template_filter(filter.get_duration, "show_duration")
app.add_template_filter(filter.get_cn_action, "show_cn_action")
