#!/usr/bin/env python3
"""
Middleware module
"""

import re

import flask

import application


def process_request(*args, **kwargs):
    """
    Middleware for authentication
    """
    path = flask.request.path
    for url in application.app.config.get("WHITE_URL"):
        if re.match(url, path):
            return None
    user = flask.session.get("user_info")
    if not user:
        return flask.redirect("/login")
    permission = user["permission"]
    if permission and flask.request.path not in permission:
        return flask.Response("not permission")
