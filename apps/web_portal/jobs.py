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

    def submit(self, job_arg, base_client_flags):
        """Submit a job through spark_submitter service."""
        entrypoint, client_flags, job_flags = self.parse_arg(job_arg)
        base_client_flags.update(client_flags)
        SparkSubmitterClient(entrypoint, base_client_flags, job_flags).submit()


class ControlProfiling(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/profiling/control/control_profiling.py'
        client_flags = {'workers': 6, 'cpu': 4, 'memory': 60}
        job_flags = {'input_data_path': job_arg.flags.get("input_data_path")}
        return (entrypoint, client_flags, job_flags)


class VehicleCalibration(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/control/calibration_table/vehicle_calibration.py'
        client_flags = {'workers': 6, 'cpu': 4, 'memory': 60}
        job_flags = {'input_data_path': job_arg.flags.get("input_data_path")}
        return (entrypoint, client_flags, job_flags)


class SensorCalibration(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/perception/sensor_calibration/calibration_multi_sensors.py'
        client_flags = {
            'workers': 2, 'cpu': 1, 'memory': 24,
            'partner_storage_writable': True,
        }
        job_flags = {'input_data_path': job_arg.flags.get("input_data_path")}
        return (entrypoint, client_flags, job_flags)


class VirtualLaneGeneration(BaseJob):
    def parse_arg(self, job_arg):
        entrypoint = 'fueling/map/generate_maps.py'
        client_flags = {
            'workers': 2, 'cpu': 1, 'memory': 24,
            'partner_storage_writable': True,
        }
        job_flags = {
            'input_data_path': job_arg.flags.get("input_data_path"),
            'output_data_path': job_arg.flags.get("output_data_path"),
            'zone_id': job_arg.flags.get("zone_id"),
            'lidar_type': job_arg.flags.get("lidar_type"),
        }
        return (entrypoint, client_flags, job_flags)
