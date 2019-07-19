#!/usr/bin/env python

"""Protobuf utils."""

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
