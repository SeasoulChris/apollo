#!/usr/bin/env python

import http

from flask_restful import Resource
import flask


class VehicleCalibration(Resource):
    """Vehicle Calibration restful service"""

    def get(self):
        return "OK"

    def post(self):
        json = flask.request.get_json()
        return json, http.HTTPStatus.ACCEPTED
