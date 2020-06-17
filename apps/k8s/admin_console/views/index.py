#!/usr/bin/env python3
"""
View functions for Index
"""

import flask


# Blueprint of index
blue_index = flask.Blueprint("index", __name__,
                             template_folder="templates",
                             static_url_path="static")


@blue_index.route('/index', methods=["GET", "POST"])
def index():
    """
    A demo function of index
    """
    return flask.render_template('index.html')
