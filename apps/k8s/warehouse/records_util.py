#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""Records utils."""

from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta
from modules.data.fuel.fueling.data.proto.record_meta_pb2 import PlanningStat


def CombineRecords(records):
    """Combine multiple records info to one."""
    records.sort(key=lambda record: record.header.begin_time)
    virtual_record = RecordMeta(path=records[0].dir, dir=records[0].dir)
    virtual_record.header.begin_time = records[0].header.begin_time
    virtual_record.header.end_time = records[-1].header.end_time

    for record in records:
        virtual_record.header.size += record.header.size
        channels = virtual_record.channels
        for channel, count in record.channels.items():
            channels[channel] = (channels.get(channel) or 0) + count

        # Set HMIStatus.
        if not virtual_record.hmi_status.current_mode:
            if record.hmi_status.current_mode:
                # A whole copy of hmi_status.
                virtual_record.hmi_status.CopyFrom(record.hmi_status)
            elif record.hmi_status.current_map:
                # This is likely a guessed map.
                virtual_record.hmi_status.current_map = record.hmi_status.current_map

        virtual_record.disengagements.extend(record.disengagements)
        virtual_record.drive_events.extend(record.drive_events)
        mileages = virtual_record.stat.mileages
        for driving_mode, miles in record.stat.mileages.items():
            mileages[driving_mode] = (mileages.get(driving_mode) or 0) + miles
        virtual_record.stat.driving_path.extend(record.stat.driving_path)

    virtual_record.planning_stat.CopyFrom(CombinePlanningMetrics(records))

    return virtual_record


def CombinePlanningMetrics(records):
    planning_stat = PlanningStat()
    if len(records) == 0:
        return planning_stat

    planning_stat.latency.max = 0
    planning_stat.latency.min = 999999
    planning_stat.latency.avg = 0

    latency_hist = {}
    weighted_avg = []
    total_weight = 0

    for record in records:
        planning_stat.latency.max = \
            max(planning_stat.latency.max, record.planning_stat.latency.max)
        planning_stat.latency.min = \
            min(planning_stat.latency.min, record.planning_stat.latency.min)

        weight = 0
        for hist_bin in record.planning_stat.latency.latency_hist:
            if hist_bin in latency_hist:
                latency_hist[hist_bin] += record.planning_stat.latency.latency_hist[hist_bin]
            else:
                latency_hist[hist_bin] = record.planning_stat.latency.latency_hist[hist_bin]
            weight += record.planning_stat.latency.latency_hist[hist_bin]
            total_weight += record.planning_stat.latency.latency_hist[hist_bin]
        weighted_avg.append(record.planning_stat.latency.avg * weight)

    avg_list = [i / float(total_weight) for i in weighted_avg]
    planning_stat.latency.avg = int(sum(avg_list))

    for key, val in latency_hist.items():
        planning_stat.latency.latency_hist[key] = val

    return planning_stat
