#!/usr/bin/env python
"""Socket related utils."""

import fueling.common.logging as logging
import sys
import os

dirs_under_runfiles = os.listdir('/home/zip_app/runfiles')
dirs_under_fuel = os.listdir('/home/zip_app/runfiles/fuel')

logging.info(F'dirs under runfiles: {dirs_under_runfiles}')
logging.info(F'dirs under fuel: {dirs_under_fuel}')

logging.info(F'sys path: {sys.path}')

import psutil
import socket



def get_ip_addr(subnet_prefix=None):
    """Get current machine's IP Address"""
    if not subnet_prefix:
        subnet_prefix = '192.168'

    # The fake subnet IP that even doesn't have to be reachable
    subnet_fake_ip = F'{subnet_prefix}.0.0'

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip = None
    try:
        sock.connect((subnet_fake_ip, 1))
        ip = sock.getsockname()[0]
        logging.info(F'got system IP address: {ip}')
    finally:
        sock.close()
    return ip


def get_socket_interface(ip):
    """Get the current socket interface name"""
    if not ip:
        return None
    addrs = psutil.net_if_addrs()
    for interface in addrs:
        network_props = addrs[interface]
        for prop in network_props:
            if str(prop).find(ip) != -1:
                logging.info(F'got system socket interface: {interface}')
                return interface
    return None

