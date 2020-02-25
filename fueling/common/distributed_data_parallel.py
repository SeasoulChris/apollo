#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
Distributed data parallel converters and utils.
"""

import os
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist

import fueling.common.logging as logging
import fueling.common.redis_utils as redis_utils
import fueling.common.socket_utils as socket_utils


class DistributedGroup(object):
    """ Distribute model, data loader or other objects to multiple machines/gpu cards"""
    is_dist_group_initialized = False
    redis_key_group_name = 'GPU.DistributedGroup'
    redis_val_world_size_name = 'world_size'
    redis_val_rank_name = 'rank'
    redis_lock_key = 'GPU.lock_key'

    @staticmethod
    def setup_group(world_size, rank):
        """Init the group, if it's not inited before"""
        if DistributedGroup.is_dist_group_initialized:
            return

        # Set backend, port and seed as consts for now
        DistributedGroup.is_dist_group_initialized = True
        backend, port, seed = 'gloo', '12355', 42
        logging.info(F'setting up group, w: {world_size}, r: {rank}')
        ip = socket_utils.get_ip_addr()
        interface = socket_utils.get_socket_interface(ip)
        if not interface:
            fatal_msg = 'unable to get socket info, fail early here to avoid uncertain status'
            logging.fatal(fatal_msg)
            raise Exception(fatal_msg)
        os.environ['MASTER_ADDR'] = ip
        os.environ['MASTER_PORT'] = port 
        os.environ['GLOO_SOCKET_IFNAME'] = interface
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.manual_seed(seed)
        logging.info('done setting up group')


    @staticmethod
    def cleanup_group():
        """Clean up the group"""
        if not DistributedGroup.is_dist_group_initialized:
            return
        logging.info('cleaning up group')
        dist.destroy_process_group()
        DistributedGroup.is_dist_group_initialized = False


def get_device_ids():
    """Get available devices"""
    return list(range(torch.cuda.device_count()))


def register_job(job_id, world_size):
    """Register the job into Redis"""
    redis_utils.redis_extend_dict(F'{DistributedGroup.redis_key_group_name}.{job_id}',
                                  {DistributedGroup.redis_val_world_size_name: world_size,
                                   DistributedGroup.redis_val_rank_name: 0})


def model_to_dist(model, world_size, job_id):
    """Convert regular model to distributed one"""
    rank = redis_utils.redis_sync_incr_dict_value(
        F'{DistributedGroup.redis_key_group_name}.{job_id}',
        DistributedGroup.redis_val_rank_name,
        F'{DistributedGroup.redis_lock_key}.{job_id}')
    if rank is None:
        logging.error('failed to sync rank, downgrade to single card mode')
        return model, 0
    rank -= 1
    DistributedGroup.setup_group(world_size, rank)
    device_ids = get_device_ids()
    model = model.to(device_ids[0])
    ddp_model = DDP(model, device_ids=device_ids)
    return ddp_model, rank


def data_loader_to_dist(dataset, batch_size, num_workers, world_size, rank):
    """
    Convert regular data loader to distributed one. The rank is from calling model_to_dist
    That says data_loader_to_dist cannot be used independently.
    """
    datasampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=datasampler,
                                       num_workers=num_workers, drop_last=True)

