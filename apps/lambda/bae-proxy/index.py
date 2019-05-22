#!/usr/bin/env python

from bae.core.wsgi import WSGIApplication
import flask
import flask_restful

from vehicle_calibration import VehicleCalibration


app = flask.Flask(__name__)
api = flask_restful.Api(app)
api.add_resource(VehicleCalibration, '/vehicle-calibration')

application = WSGIApplication(app)
