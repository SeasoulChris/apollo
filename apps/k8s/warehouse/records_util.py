#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""Records utils."""

from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta


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

        if record.hmi_status.current_mode:
            virtual_record.hmi_status.CopyFrom(record.hmi_status)

        virtual_record.disengagements.extend(record.disengagements)
        virtual_record.drive_events.extend(record.drive_events)
        mileages = virtual_record.stat.mileages
        for driving_mode, miles in record.stat.mileages.items():
            mileages[driving_mode] = (mileages.get(driving_mode) or 0) + miles
        virtual_record.stat.driving_path.extend(record.stat.driving_path)
    return virtual_record
