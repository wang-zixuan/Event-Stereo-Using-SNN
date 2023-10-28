import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # datasets
    if opt['datasets'].get('dataroot') is not None:
        opt['datasets']['dataroot'] = osp.expanduser(opt['datasets']['dataroot'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    opt['path']['root'] = "/data1/wzx/results"

    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'eccv_exp', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['tb_logger'] = osp.join(experiments_root, 'tb_logger')
        # TODO
        opt['path']['log'] = experiments_root

    else:  # test
        results_root = osp.join(opt['path']['root'], 'test_result', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['tb_logger'] = osp.join(results_root, 'tb_logger')

    return opt
