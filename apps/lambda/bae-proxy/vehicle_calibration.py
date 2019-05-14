#!/usr/bin/env python

from flask_restful import Resource


class VehicleCalibration(Resource):
    def get(self):
        return {'hello': 'world'}
