#!/usr/bin/env python
"""Protobuf utils."""

import google.protobuf.json_format as json_format
import google.protobuf.text_format as text_format


def write_pb_to_text_file(topic_pb, file_path):
    """write pb message to file"""
    with open(file_path, 'w') as f:
        f.write(str(topic_pb))


def get_pb_from_text_file(filename, pb_value):
    """Get a proto from given text file."""
    with open(filename, 'r') as file_in:
        return text_format.Merge(file_in.read(), pb_value)


def get_pb_from_bin_file(filename, pb_value):
    """Get a proto from given binary file."""
    with open(filename, 'rb') as file_in:
        pb_value.ParseFromString(file_in.read())
    return pb_value


def get_pb_from_file(filename, pb_value):
    """Get a proto from given file by trying binary mode and text mode."""
    try:
        return get_pb_from_bin_file(filename, pb_value)
    except text_format.ParseError:
        try:
            return get_pb_from_text_file(filename, pb_value)
        except text_format.ParseError:
            print('Error: Cannot parse {} as binary or text proto'.format(filename))

    return None


def pb_to_dict(pb, include_default_value_fields=False, preserve_proto_field_name=True):
    """Convert proto to dict."""
    return json_format.MessageToDict(pb, include_default_value_fields, preserve_proto_field_name)


def dict_to_pb(input_dict, pb_value, ignore_unknown_fields=True):
    """Convert dict to proto."""
    return json_format.ParseDict(input_dict, pb_value, ignore_unknown_fields)
