import torch
import numpy as np
import random
from utils.logger import get_root_logger
from functools import partial
from utils.dist_util import get_dist_info
from data.prefetch_dataloader import PrefetchDataLoader


def create_dataloader(dataset,
                      dataset_opt,
                      num_gpu=1,
                      dist=False,
                      seed=None,
                      sampler=None,
                      is_train=True):
    multiplier = 1 if num_gpu == 0 else num_gpu
    batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
    num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
    if is_train:
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=False,
            worker_init_fn=seed_worker
        )

        if sampler is None:
            dataloader_args['shuffle'] = True

    else:  # validation
        dataloader_args = dict(
            dataset=dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, 
            drop_last=False)

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', True)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: '
                    f'num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        return torch.utils.data.DataLoader(**dataloader_args)


def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)
