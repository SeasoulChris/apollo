#!/usr/bin/env python

from apps.k8s.spark_submitter.client import SparkSubmitterClient


class BaseJob(object):
    def parse_arg(self, job_arg):
        """
        job_arg: An instance of input SaasJobArg.

        @return a tuple of (entrypoint, client_flags, job_flags).
        entrypoint: Job entrypoint.
        client_flags: A dict of flags to control SparkSubmitterClient.
        job_flags: A dict of flags to control SparkSubmitterClient.
        """
        raise Exception('Not implemented!')

    def submit(self, job_arg, base_client_flags, base_job_flags):
        """Submit a job through spark_submitter service."""
        entrypoint, client_flags, job_flags = self.parse_arg(job_arg)
        client_flags.update(base_client_flags)
        job_flags.update(base_job_flags)
        return SparkSubmitterClient(entrypoint, client_flags, job_flags).submit_via_call()


class ControlProfiling(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/profiling/control/control_profiling.py'
        client_flags = {'workers': 6, 'cpu': 4, 'memory': 60}
        job_flags = {'input_data_path': job_arg.flags.get("input_data_path")}
        return (entrypoint, client_flags, job_flags)


class OpenSpacePlannerProfiling(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/profiling/open_space_planner/metrics.py'
        client_flags = {
            'workers': 4, 'cpu': 1, 'disk': 1, 'memory': 1,
            'partner_storage_writable': True,
        }
        job_flags = {
            'input_data_path': job_arg.flags.get("input_data_path"),
            'output_data_path': job_arg.flags.get("output_data_path"),
            'open_space_planner_profiling_generate_report': True,
        }
        return (entrypoint, client_flags, job_flags)


class VehicleCalibration(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/control/calibration_table/vehicle_calibration.py'
        client_flags = {'workers': 6, 'cpu': 4, 'memory': 60}
        job_flags = {
            'input_data_path': job_arg.flags.get("input_data_path"),
        }
        return (entrypoint, client_flags, job_flags)


class SensorCalibration(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/perception/sensor_calibration/calibration_multi_sensors.py'
        client_flags = {
            'workers': 4, 'cpu': 4, 'memory': 32,
            'partner_storage_writable': True,
        }
        job_flags = {
            'input_data_path': job_arg.flags.get("input_data_path"),
            'output_data_path': job_arg.flags.get("output_data_path"),
        }
        return (entrypoint, client_flags, job_flags)


class VirtualLaneGeneration(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/map/generate_maps.py'
        client_flags = {
            'workers': 4, 'cpu': 4, 'memory': 64,
            'partner_storage_writable': True,
        }
        job_flags = {
            'input_data_path': job_arg.flags.get("input_data_path"),
            'output_data_path': job_arg.flags.get("output_data_path"),
            'zone_id': job_arg.flags.get("zone_id"),
            'lidar_type': job_arg.flags.get("lidar_type"),
            'lane_width': job_arg.flags.get("lane_width"),
            'extra_roi_extension': job_arg.flags.get("extra_roi_extension"),
        }
        return (entrypoint, client_flags, job_flags)


class PredictionModelTraining(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/prediction/prediction_end_to_end.py'
        client_flags = {
            'workers': 1, 'gpu': 1, 'memory': 90, 'disk': 200,
            'partner_storage_writable': True,
        }
        job_flags = {
            'input_path': job_arg.flags.get("input_data_path"),
        }
        return (entrypoint, client_flags, job_flags)
