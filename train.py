import torch
import argparse
import random
import logging
import math
import time
import datetime
import os
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from data.data_sampler import EnlargedSampler
import torch.multiprocessing as mp
from utils.dist_util import get_dist_info, init_dist
from os import path as osp
from utils.options import parse
from utils.misc import set_random_seed, make_exp_dirs, get_time_str
from utils.logger import get_root_logger, get_env_info, dict2str, init_tb_logger, MessageLogger
from data import dataset, dataset_constants
from data.dataloader_utils import create_dataloader
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models.stereo_model import StereoNet
from data.transformers import init_transformers


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default='options/train_unsupervised_spikingjelly.yml', help='Path to option YAML file.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    opt['dist'] = False
    opt['rank'], opt['world_size'] = get_dist_info()

    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 1000)
        opt['manual_seed'] = seed
    set_random_seed(seed)

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='event_stereo', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = init_tb_logger(log_dir=opt['path']['tb_logger'])
    return logger, tb_logger


def create_dataset_loader(opt, logger):
    dataset_transformers = init_transformers(slow_split=opt['datasets']['slow_split'], fast_split=opt['datasets']['fast_split'])

    train_set, val_set, test_set = dataset.IndoorFlying.split(opt['datasets']['indoor_flying_dataroot'], split_number=opt['datasets']['split_number'], transformers=dataset_transformers)

    torch.save(train_set, f'/trainset_lif_{opt["datasets"]["split_number"]}.pt')
    torch.save(test_set, f'/testset_lif_{opt["datasets"]["split_number"]}.pt')

    train_set = torch.load(f'/data1/wzx/trainset_lif_{opt["datasets"]["split_number"]}.pt')
    test_set = torch.load(f'/data1/wzx/testset_lif_{opt["datasets"]["split_number"]}.pt')

    # train_set = torch.load(f'/trainset_hed_binary_dense_{opt["datasets"]["split_number"]}.pt')
    # test_set = torch.load(f'/testset_hed_binary_dense_{opt["datasets"]["split_number"]}.pt')

    # outdoor_day_train_set = dataset.OutdoorDay.get_train_dataset(opt['datasets']['outdoor_day_dataroot'], opt['datasets']['time_horizon'])

    # train_set = torch.utils.data.ConcatDataset((indoor_flying_train_set, outdoor_day_train_set))

    dataset_enlarge_ratio = opt['datasets'].get('dataset_enlarge_ratio', 1)
    train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)

    train_set_loader = create_dataloader(train_set, opt['datasets'], num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, is_train=True, seed=opt['manual_seed'])
    test_set_loader = create_dataloader(test_set, opt['datasets'], num_gpu=opt['num_gpu'], is_train=False)

    logger.info('Dataset created.')

    num_iter_per_epoch = math.ceil(
        len(train_set) * dataset_enlarge_ratio / (opt['datasets']['batch_size_per_gpu'] * opt['num_gpu']))

    total_epochs = int(opt['train']['total_epochs'])
    total_iters = total_epochs * num_iter_per_epoch
    opt['train']['total_iter'] = total_iters

    logger.info(
        'Statistics:'
        f'\n\tSplit number: {opt["datasets"]["split_number"]}'
        f'\n\tNumber of train seqs: {len(train_set)}'
        # f'\n\tNumber of val seqs: {len(val_set)}'
        f'\n\tNumber of test seqs: {len(test_set)}'
        f'\n\tLength of train loader per gpu: {len(train_set_loader)}'
        f'\n\tBatch size per gpu: {opt["datasets"]["batch_size_per_gpu"]}'
        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.'
    )

    return train_set_loader, test_set_loader, train_sampler, total_epochs


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)

    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    if resume_state is None:
        make_exp_dirs(opt)

    logger, tb_logger = init_loggers(opt)

    train_loader, test_loader, train_sampler, total_epochs = create_dataset_loader(opt, logger)
    if resume_state:  # resume training
        model = StereoNet(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = StereoNet(opt)
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    training_losses = []
    mdes = []

    for epoch in range(start_epoch, total_epochs):
        mean_depth_error, one_pixel_error, mean_disparity_error = 0, 0, 0
        train_sampler.set_epoch(epoch)

        for train_data in train_loader:
            data_time = time.time() - data_time
            
            current_iter += 1

            model.feed_data(train_data)
            model.optimize_parameters()

            mean_depth_error += model.log_dict['mean_depth_error']
            one_pixel_error += model.log_dict['one_pixel_error']
            mean_disparity_error += model.log_dict['mean_disparity_error']

            training_losses.append(model.log_dict['l_pix'])
            mdes.append(model.log_dict['mean_disparity_error'])

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            data_time = time.time()
            iter_time = time.time()
        
        model.update_learning_rate()

        mean_depth_error /= len(train_loader)
        one_pixel_error /= len(train_loader)
        mean_disparity_error /= len(train_loader)

        logger.info(f'Training mean_depth_error: {mean_depth_error * 100}cm, 1PA: {100 - one_pixel_error}%, mean disparity error: {mean_disparity_error}pix.')

        logger.info('Saving models and training states.')
        model.save(epoch, current_iter)

        logger.info('testing...')
        model.validate(test_loader, epoch)
        logger.info('testing done.')

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    logger.info(f'best epoch: {model.best_epoch}, mde: {model.min_mde * 100}cm')
    tb_logger.close()


def create_testset_loader(opt, logger):
    test_set = torch.load(f'/testset_hed_binary_dense_{opt["datasets"]["split_number"]}.pt')
    test_set_loader = create_dataloader(test_set, opt['datasets'], num_gpu=opt['num_gpu'], is_train=False)

    logger.info('Dataset created.')
    logger.info(
        'Statistics:'
        f'\n\tSplit number: {opt["datasets"]["split_number"]}'
        f'\n\tNumber of test seqs: {len(test_set)}'
    )

    return test_set_loader


def main_test(rank=None, world_size=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    opt = parse_options(is_train=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    make_exp_dirs(opt)
    logger, _ = init_loggers(opt)

    test_loader = create_testset_loader(opt, logger)

    model = StereoNet(opt, rank)

    logger.info('testing...')
    model.test(test_loader)
    logger.info('testing done.')


if __name__ == '__main__':
    main()
    # main_test()
