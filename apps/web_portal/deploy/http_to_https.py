#!/usr/bin/env python
"""HTTP to HTTPS auto redirect."""

import flask


app = flask.Flask(__name__)


@app.route('/')
@app.route('/<path:request_path>')
def index(request_path=''):
    return flask.redirect(flask.request.url.replace('http://', 'https://', 1))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
