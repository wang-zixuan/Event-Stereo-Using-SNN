import importlib
from tkinter import E
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import os
from models import errors
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.logger import get_root_logger
from utils.image_utils import disparity_to_depth, depth_to_disparity
from spikingjelly.clock_driven import functional, surrogate, neuron, layer
import matplotlib.pyplot as plt
import matplotlib
import cv2
from torchvision.models import vgg as vgg
import numpy as np
matplotlib.use('Agg')

logger = logging.getLogger('event_stereo')
loss_module = importlib.import_module('models.losses')
network_module = importlib.import_module('models.stereo_matching_snn')

VGG_PRETRAIN_PATH = {
    'vgg11': '/data1/wzx/results/vgg/vgg11-bbd30ac9.pth',
    'vgg13': '/data1/wzx/results/vgg/vgg13-c768596a.pth',
    'vgg16': '/data1/wzx/results/vgg/vgg16-397923af.pth',
    'vgg19': '/data1/wzx/results/vgg/vgg19-dcbb9e9d.pth',
}

NAMES = {   
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
        'pool5'
    ]
}


class StereoNet:
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt['is_train']
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

        self.best_epoch = -1
        self.min_mde = float('inf')

        network_type = self.opt['train']['network'].pop('type')
        network_cls = getattr(network_module, network_type)
        self.network = network_cls(**self.opt['train']['network']).to(self.device)

        self.print_network(self.network)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.network, load_path, self.opt['path'].get('strict_load_g', True))

        if self.opt['num_gpu'] > 1:
            self.network = DataParallel(self.network)

        if self.is_train:
            self.init_training_settings()
        
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=NAMES['vgg11'], vgg_type='vgg11').to(self.device)

    def init_training_settings(self):
        self.network.train()
        train_opt = self.opt['train']

        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('mask_opt'):
            mask_type = train_opt['mask_opt'].pop('type')
            cri_mask_cls = getattr(loss_module, mask_type)
            self.cri_mask = cri_mask_cls(**train_opt['mask_opt']).to(self.device)
        else:
            self.cri_mask = None

        if train_opt.get('image_opt'):
            image_type = train_opt['image_opt'].pop('type')
            cri_image_cls = getattr(loss_module, image_type)
            self.cri_image = cri_image_cls(**train_opt['image_opt']).to(self.device)
        else:
            self.cri_image = None

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim'].pop('type')
        if optim_type == 'Adam':
            self.optimizer = torch.optim.Adam(optim_params, **train_opt['optim'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')

    def setup_schedulers(self):
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type == 'MultiStepLR':
            self.scheduler = \
                torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **train_opt['scheduler'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def print_network(self, net):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module

        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def feed_data(self, data):
        self.left_edge_map = data['left']['edge_map'].to(self.device)
        self.frame_index = data['frame_index'].to(self.device)

        self.left_seq_slow, self.right_seq_slow = data['left']['cur_idx_event_sequence_slow'].to(self.device), data['right']['cur_idx_event_sequence_slow'].to(self.device)

        self.left_image = data['left']['image'].unsqueeze(0).to(self.device)
        
        self.depth_image_dense = data['left']['depth_image_gt'].to(self.device)

    def optimize_parameters(self):
        functional.reset_net(self.network)

        self.optimizer.zero_grad()

        network_input_slow = torch.cat((self.left_seq_slow, self.right_seq_slow), dim=2)

        depth, mask, _, pred_image, _ = self.network(network_input_slow)

        network_input_1ch = torch.sum(network_input_slow.squeeze(0), dim=1).expand(1, 3, 260, 346)

        loss_dict = OrderedDict()

        l_total = 0

        if self.cri_pix:
            l_pix = self.cri_pix(depth, self.depth_image_dense)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_mask:
            l_mask = self.cri_mask(mask, self.left_edge_map, self.depth_image_dense)
            l_total += l_mask
            loss_dict['l_mask'] = l_mask
        
        if self.cri_image:
            l_image = self.cri_image(pred_image, network_input_1ch)
            l_total += l_image
            loss_dict['l_image'] = l_image

        l_total.backward()
        self.optimizer.step()
        self.network.detach()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.compute_error(depth[0], self.depth_image_dense)

    def reduce_loss_dict(self, loss_dict):
        with torch.no_grad():
            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def update_learning_rate(self):
        self.scheduler.step()

    def compute_error(self, estimated_depth, ground_truth_depth, is_train=True):
        estimated_disparity = depth_to_disparity(estimated_depth).squeeze(0)
        ground_truth_disparity = depth_to_disparity(ground_truth_depth).squeeze(0)

        mean_depth_error = errors.compute_absolute_error(estimated_depth, ground_truth_depth)[1]
        binary_error_map, one_pixel_error = errors.compute_n_pixels_error(estimated_disparity, ground_truth_disparity, n=1.0)
        mean_disparity_error = errors.compute_absolute_error(estimated_disparity, ground_truth_disparity)[1]

        if is_train:
            self.log_dict['mean_depth_error'] = mean_depth_error
            self.log_dict['one_pixel_error'] = one_pixel_error
            self.log_dict['mean_disparity_error'] = mean_disparity_error
        else:
            return mean_depth_error, one_pixel_error, mean_disparity_error

    def get_current_learning_rate(self):
        return [self.optimizer.state_dict()['param_groups'][0]['lr']]

    def get_current_log(self):
        return self.log_dict

    def save(self, epoch, current_iter):
        self.save_network(self.network, 'net', epoch, current_iter)
        self.save_training_state(epoch, current_iter)

    def save_network(self, net, net_label, epoch, current_iter, param_key='params'):
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{epoch}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            if isinstance(net_, (DataParallel, DistributedDataParallel)):
                net_ = net_.module
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)

    def save_training_state(self, epoch, current_iter):
        if current_iter == -1:
            current_iter = 'latest'

        state = {
            'epoch': epoch,
            'iter': current_iter,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        save_filename = f'{epoch}_{current_iter}.state'
        save_path = os.path.join(self.opt['path']['training_states'], save_filename)
        # torch.save(state, save_path)

    def resume_training(self, resume_state):
        resume_optimizer = resume_state['optimizer']
        resume_scheduler = resume_state['scheduler']
        self.optimizer.load_state_dict(resume_optimizer)
        self.scheduler.load_state_dict(resume_scheduler)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

    def validate(self, test_loader, epoch):
        self.network.eval()
        with torch.no_grad():
            mean_depth_error, one_pixel_error, mean_disparity_error = 0, 0, 0
            encoder_spkr, decoder_spkr, btn_spkr, epm_spkr = 0, 0, 0, 0
            for batch in test_loader:
                self.feed_data(batch) 
                network_input_slow = torch.cat((self.left_seq_slow, self.right_seq_slow), dim=2)
                functional.reset_net(self.network)
                output, edge_mask, v, img_feature, _ = self.network(network_input_slow)
                # self.network.detach()

                cur_mde, cur_1pe, cur_mdispe = self.compute_error(output[0], self.depth_image_dense, is_train=False)

                mean_depth_error += cur_mde
                one_pixel_error += cur_1pe
                mean_disparity_error += cur_mdispe
                
                # spiking rate 
                functional.reset_net(self.network)
                dict_spkr = self.network.calculate_firing_rates(network_input_slow)
                
                encoder_spkr += dict_spkr['encoder']
                decoder_spkr += dict_spkr['decoder']
                btn_spkr += dict_spkr['bottleneck']

                if 'EPM' in dict_spkr.keys():
                    epm_spkr += dict_spkr['EPM']

        mean_depth_error /= len(test_loader)
        one_pixel_error /= len(test_loader)
        mean_disparity_error /= len(test_loader)

        if mean_depth_error < self.min_mde:
            self.min_mde = mean_depth_error
            self.best_epoch = epoch

        encoder_spkr /= len(test_loader)
        decoder_spkr /= len(test_loader)
        btn_spkr /= len(test_loader)
        epm_spkr /= len(test_loader)

        logger.info(f'encoder spk: {encoder_spkr}, decoder: {decoder_spkr}, btn: {btn_spkr}, epm: {epm_spkr}')

        logger.info(f'Testing mean_depth_error: {mean_depth_error * 100}cm, 1PA: {100 - one_pixel_error}%, mean disparity error: {mean_disparity_error}pix.')

        self.network.train()

    def test(self, test_loader):
        self.network.eval()
        with torch.no_grad():
            mean_depth_error, one_pixel_error, mean_disparity_error = 0, 0, 0
            encoder_spkr, decoder_spkr, btn_spkr, epm_spkr = 0, 0, 0, 0
            flops = 0
            for idx, batch in enumerate(test_loader):
                self.feed_data(batch)

                if int(self.frame_index) != 1057: continue

                save_path_v = os.path.join(self.opt['path']['results_root'], f's2_{int(self.frame_index)}_v.png')

                network_input_slow = torch.cat((self.left_seq_slow, self.right_seq_slow), dim=2)
                functional.reset_net(self.network)
                # flops += self.network.calculate_computation(network_input_slow) / len(test_loader)

                output, edge_mask, v, img_feature, out_spks = self.network(network_input_slow)
                # self.network.detach()

                plt.figure()
                plt.imshow(v[0].squeeze(0).squeeze(0).cpu().numpy(), cmap='magma')
                plt.axis('off')
                plt.margins(0, 0)
                # plt.clim(0, 5)
                plt.savefig(save_path_v, bbox_inches='tight', dpi=600, pad_inches=0.0)
                plt.close()

        mean_depth_error /= len(test_loader)
        one_pixel_error /= len(test_loader)
        mean_disparity_error /= len(test_loader)

        encoder_spkr /= len(test_loader)
        decoder_spkr /= len(test_loader)
        btn_spkr /= len(test_loader)
        epm_spkr /= len(test_loader)

        logger.info(f'flops: {flops}')

        logger.info(f'encoder spk: {encoder_spkr}, decoder: {decoder_spkr}, btn: {btn_spkr}, epm: {epm_spkr}')

        logger.info(f'Testing mean_depth_error: {mean_depth_error * 100}cm, 1PA: {100 - one_pixel_error}%, mean disparity error: {mean_disparity_error}pix.')


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace('_bn', '')]

        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(VGG_PRETRAIN_PATH[vgg_type]):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(
                VGG_PRETRAIN_PATH[vgg_type], map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(
                        kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        
        return output