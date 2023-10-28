import os
import math
import torch
import functools
from torch.nn import functional
from collections import OrderedDict
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models import vgg as vgg
from torchvision.models import resnet as resnet

_reduction_modes = ['none', 'mean', 'sum']
VGG_PRETRAIN_PATH = {
    'vgg11': '/vgg/vgg11-bbd30ac9.pth',
    'vgg13': '/vgg/vgg13-c768596a.pth',
    'vgg16': '/vgg/vgg16-397923af.pth',
    'vgg19': '/vgg/vgg19-dcbb9e9d.pth',
}

RESNET_PRETRAIN_PATH = {
    'resnet18': '/resnet/resnet18-5c106cde.pth',
    'resnet34': '/resnet/resnet34-333f7ec4.pth',
    'resnet50': '/resnet/resnet50-19c8e357.pth',
    'resnet101': '/resnet/resnet101-5d3b4d8f.pth',
    'resnet152': '/resnet/resnet152-b121ed2d.pth'
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

RESNET_NAMES = {
    'resnet18': [
        'conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'layer4'
    ],
    'resnet34': [
        'conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'layer4'
    ],
    'resnet50': [
        'conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'layer4'
    ],
    'resnet101': [
        'conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'layer4'
    ],
    'resnet152': [
        'conv1', 'bn1', 'relu', 'layer1', 'layer2', 'layer3', 'layer4'
    ]
}


def ScaleInvariant_Loss(predicted, groundtruth):
    """
    Referred to as 'scale-invariant loss' in the paper 'learning monocular dense depth from events' (3DV 2020)
    See also 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos'

    :param predicted:
    :param groundtruth:
    :return:
    """
    mask = ~torch.isnan(groundtruth)
    n = torch.sum(mask != False)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res[mask == False] = 0  # invalid pixels: nan --> 0

    MSE = torch.sum(torch.pow(res[mask], 2)) / n
    quad = torch.pow(torch.sum(res[mask]), 2) / (n ** 2) 

    return MSE - quad


def Multiscale_ScaleInvariant_Loss(predicted, groundtruth, factors=(1., 1., 1., 1.)):
    """
    :param predicted: a tuple of num_scales [N, 1, H, W] tensors
    :param groundtruth: a tuple of num_scales [N, 1, H, W] tensors
    :return: a scalar value
    """
    multiscale_loss = 0.0

    for (factor, map) in zip(factors, predicted):
        multiscale_loss += factor * ScaleInvariant_Loss(map, groundtruth)

    return multiscale_loss


def GradientMatching_Loss(predicted, groundtruth):
    """
    Referred to as 'multi-scale scale-invariant gradient matching loss' in the paper 'learning monocular dense depth from events' (3DV 2020)
    See also 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos'

    :param predicted:
    :param groundtruth:
    :return:
    """
    mask = ~torch.isnan(groundtruth)
    n = torch.sum(mask != False)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res[mask == False] = 0  # invalid pixels: nan --> 0

    # define sobel filters for each direction
    if torch.cuda.is_available():
        sobelX = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3)).cuda()
        sobelY = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3)).cuda()
    else:
        sobelX = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1, 1, 3, 3))
        sobelY = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view((1, 1, 3, 3))

    # stride and padding of 1 to keep the same resolution
    grad_res_x = F.conv2d(res, sobelX, stride=1, padding=1)
    grad_res_y = F.conv2d(res, sobelY, stride=1, padding=1)

    # use the value of the gradients only at valid pixel locations
    grad_res_x *= mask
    grad_res_y *= mask

    return torch.sum(torch.abs(grad_res_x[mask]) + torch.abs(grad_res_y[mask])) / n


def MultiScale_GradientMatching_Loss(predicted, groundtruth, factors=(1., 1., 1., 1.)):
    """
    Computes the gradient matching loss at each scale, then return the sum.

    :param predicted: a tuple of num_scales [N, 1, H, W] tensors
    :param groundtruth: a tuple of num_scales [N, 1, H, W] tensors
    :return: a scalar value
    """
    multiscale_loss = 0.0

    for (factor, map) in zip(factors, predicted):
        multiscale_loss += factor * GradientMatching_Loss(map, groundtruth)

    return multiscale_loss


def SpikePenalization_Loss(intermediary_spike_tensors):
    """
    Regularization loss to diminish the spiking activity of the network. Penalizes the square of the mean spike counts.

    :param intermediary_spike_tensors: a list of integer spike tensors
    """
    spk_penalization_loss = 0.0

    for spike_tensor in intermediary_spike_tensors:
        spk_penalization_loss += 1 / (2 * spike_tensor.numel()) * torch.sum(torch.pow(spike_tensor, 2))

    return spk_penalization_loss


class Total_Loss(nn.Module):
    """
    For learning linear (metric) depth, use alpha=0.5
    Tests were done without any weighting of predictions at different scales --> scale_weights = (1., 1., 1., 1.)

    Spike penalization can be balanced with beta weight parameter. Increasing it will reduce spiking activity and accuracy.
    """

    def __init__(self, alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=False, beta=1.):
        super(Total_Loss, self).__init__()
        self.alpha = alpha
        self.scale_weights = scale_weights
        self.penalize_spikes = penalize_spikes
        self.beta = beta

    def forward(self, predicted, groundtruth):
        return Multiscale_ScaleInvariant_Loss(predicted, groundtruth, self.scale_weights) + self.alpha * MultiScale_GradientMatching_Loss(predicted, groundtruth, self.scale_weights)


class SizeAdapter:
    def __init__(self, minimum_size=64):
        self.minimum_size = minimum_size
        self.pixels_pad_to_width = None
        self.pixels_pad_to_height = None

    def closest_larger_multiple_of_minimum_size(self, size):
        return int(math.ceil(size / self.minimum_size) * self.minimum_size)

    def pad(self, network_input):
        height, width = network_input.size()[-2:]
        self.pixels_pad_to_height = (
            self.closest_larger_multiple_of_minimum_size(height) - height)
        self.pixels_pad_to_width = (
            self.closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self.pixels_pad_to_width, 0,
                             self.pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        return network_output[..., self.pixels_pad_to_height:, self.pixels_pad_to_width:]


class StyleLoss_vgg11(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_vgg11, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=NAMES['vgg11'], vgg_type='vgg11')

    def forward(self, x, gt):
        vgg_features = self.vgg_extractor(gt / torch.max(gt))
        gt_feature = [vgg_features['pool1'], vgg_features['pool2'], vgg_features['pool3'], vgg_features['pool4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_vgg13(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_vgg13, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=NAMES['vgg13'], vgg_type='vgg13')

    def forward(self, x, gt):
        vgg_features = self.vgg_extractor(gt / 255.)
        gt_feature = [vgg_features['pool1'], vgg_features['pool2'], vgg_features['pool3'], vgg_features['pool4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_vgg16(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_vgg16, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=NAMES['vgg16'], vgg_type='vgg16')

    def forward(self, x, gt):
        vgg_features = self.vgg_extractor(gt / 255.)
        gt_feature = [vgg_features['pool1'], vgg_features['pool2'], vgg_features['pool3'], vgg_features['pool4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_vgg19(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_vgg19, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=NAMES['vgg19'], vgg_type='vgg19')

    def forward(self, x, gt):
        vgg_features = self.vgg_extractor(gt / 255.)
        gt_feature = [vgg_features['pool1'], vgg_features['pool2'], vgg_features['pool3'], vgg_features['pool4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_resnet18(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_resnet18, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.resnet_extractor = ResnetFeatureExtractor(layer_name_list=RESNET_NAMES['resnet18'], resnet_type='resnet18')

    def forward(self, x, gt):
        resnet_features = self.resnet_extractor(gt / 255.)
        gt_feature = [resnet_features['layer1'], resnet_features['layer2'], resnet_features['layer3'], resnet_features['layer4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_resnet34(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_resnet34, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.resnet_extractor = ResnetFeatureExtractor(layer_name_list=RESNET_NAMES['resnet34'], resnet_type='resnet34')

    def forward(self, x, gt):
        resnet_features = self.resnet_extractor(gt / 255.)
        gt_feature = [resnet_features['layer1'], resnet_features['layer2'], resnet_features['layer3'], resnet_features['layer4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_resnet50(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_resnet50, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.resnet_extractor = ResnetFeatureExtractor(layer_name_list=RESNET_NAMES['resnet50'], resnet_type='resnet50')

    def forward(self, x, gt):
        resnet_features = self.resnet_extractor(gt / 255.)
        gt_feature = [resnet_features['layer1'], resnet_features['layer2'], resnet_features['layer3'], resnet_features['layer4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_resnet101(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_resnet101, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.resnet_extractor = ResnetFeatureExtractor(layer_name_list=RESNET_NAMES['resnet101'], resnet_type='resnet101')

    def forward(self, x, gt):
        resnet_features = self.resnet_extractor(gt / 255.)
        gt_feature = [resnet_features['layer1'], resnet_features['layer2'], resnet_features['layer3'], resnet_features['layer4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class StyleLoss_resnet152(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(StyleLoss_resnet152, self).__init__()
        self.loss_weight = loss_weight
        # self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.resnet_extractor = ResnetFeatureExtractor(layer_name_list=RESNET_NAMES['resnet152'], resnet_type='resnet152')

    def forward(self, x, gt):
        resnet_features = self.resnet_extractor(gt / 255.)
        gt_feature = [resnet_features['pool1'], resnet_features['pool2'], resnet_features['pool3'], resnet_features['pool4']]

        style_loss = 0
        for i, neuron_v in enumerate(x):
            style_loss += self.criterion(neuron_v, gt_feature[i]) * self.loss_weight
        return style_loss


class ResnetFeatureExtractor(nn.Module):
    def __init__(self,
                 layer_name_list,
                 resnet_type='resnet18',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False):
        super(ResnetFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if os.path.exists(RESNET_PRETRAIN_PATH[resnet_type]):
            resnet_net = getattr(resnet, resnet_type)(pretrained=False)
            state_dict = torch.load(
                RESNET_PRETRAIN_PATH[resnet_type], map_location=lambda storage, loc: storage)
            resnet_net.load_state_dict(state_dict)
        else:
            resnet_net = getattr(resnet, resnet_type)(pretrained=True)

        named_modules_resnet = dict(resnet_net.named_modules())

        modified_net = OrderedDict()
        for k, v in named_modules_resnet.items():
            if k not in self.layer_name_list:
                continue
            modified_net[k] = v

        self.resnet_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.resnet_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.resnet_net.train()
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
        for key, layer in self.resnet_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        
        return output


class MaskLoss_CE_All_binary(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskLoss_CE_All_binary, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_weight = loss_weight

    def forward(self, predicted_mask, gt_mask, depth_image):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        gt_mask = gt_mask[mask]

        points_have_edge_sum = torch.sum(gt_mask)
        pos_weight = (n - points_have_edge_sum) / points_have_edge_sum

        for pred in predicted_mask:
            pred = pred[mask]

            pred[pred == 0] = -1

            loss += torch.sum(pos_weight * gt_mask * (-torch.log(self.sigmoid(pred))) + (1 - gt_mask) * (-torch.log(1 - self.sigmoid(pred)))) / n

        return self.loss_weight * loss


class MaskLoss_CE_All_binary_5(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskLoss_CE_All_binary_5, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_weight = loss_weight

    def forward(self, predicted_mask, gt_mask, depth_image):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        gt_mask = gt_mask[mask]

        points_have_edge_sum = torch.sum(gt_mask)
        pos_weight = (n - points_have_edge_sum) / points_have_edge_sum

        for pred in predicted_mask:
            pred = pred[mask]
            pred[pred == 1] = 5
            pred[pred == 0] = -5

            loss += torch.sum(pos_weight * gt_mask * (-torch.log(self.sigmoid(pred))) + (1 - gt_mask) * (-torch.log(1 - self.sigmoid(pred)))) / n

        return self.loss_weight * loss


class MaskLoss_CE_All_binary_weight(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskLoss_CE_All_binary_weight, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_weight = loss_weight

    def forward(self, predicted_mask, gt_mask, depth_image):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        gt_mask = gt_mask[mask]

        points_have_edge_sum = torch.sum(gt_mask)
        points_not_have_edge_sum = n - points_have_edge_sum

        pos_weight = points_not_have_edge_sum / n 
        neg_weight = points_have_edge_sum / n

        for pred in predicted_mask:
            pred = pred[mask]
            pred[pred == 0] = -1

            loss += torch.sum(pos_weight * gt_mask * (-torch.log(self.sigmoid(pred))) + neg_weight * (1 - gt_mask) * (-torch.log(1 - self.sigmoid(pred)))) / n

        return self.loss_weight * loss


class MaskLoss_CE_Sigmoid(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskLoss_CE_Sigmoid, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_weight = loss_weight

    def forward(self, predicted_mask, gt_mask, depth_image):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        gt_mask = gt_mask[mask]

        points_have_edge_sum = torch.sum(gt_mask)
        pos_weight = (n - points_have_edge_sum) / points_have_edge_sum
        # print('loss pos weight', pos_weight)

        for pred in predicted_mask:
            pred = pred[mask]
            loss += torch.sum(pos_weight * gt_mask * (-torch.log(self.sigmoid(pred))) + (1 - gt_mask) * (-torch.log(1 - self.sigmoid(pred)))) / n

        return self.loss_weight * loss


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, predicted_mask, gt_mask):
        loss = 0
        for mask in predicted_mask:
            loss += l1_loss(mask, gt_mask)

        return loss


class MaskLoss_CE_All(nn.Module):
    def __init__(self):
        super(MaskLoss_CE_All, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, predicted_mask, gt_mask, depth_image, pos_weight=None):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        gt_mask = gt_mask[mask]

        binary_gt_mask = gt_mask.clone()
        binary_gt_mask[binary_gt_mask != 0] = 1.

        points_have_edge_sum = torch.sum(binary_gt_mask)
        pos_weight = (n - points_have_edge_sum) / points_have_edge_sum

        for pred in predicted_mask:
            pred = pred[mask]

            loss += torch.sum(pos_weight * binary_gt_mask * (-torch.log(self.sigmoid(pred))) + (1 - binary_gt_mask) * (-torch.log(1 - self.sigmoid(pred)))) / n

        return loss


class MaskLoss_CE_All_weight(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskLoss_CE_All_weight, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_weight = loss_weight

    def forward(self, predicted_mask, gt_mask, depth_image):
        loss = 0
        mask = ~torch.isnan(depth_image)  # depth image 为 nan 的地方不参与 loss 计算
        n = torch.sum(mask != False)  # 有效点的个数
        gt_mask = gt_mask[mask]  # 对 gt 进行 mask

        binary_gt_mask = gt_mask.clone()  # 对 gt 进行二值化处理
        binary_gt_mask[binary_gt_mask != 0] = 1.

        # points_have_edge_sum = torch.sum(binary_gt_mask)  # 计算 gt 中有 edge 的点数和
        # pos_weight = 1.0 * points_have_edge_sum / n
        # neg_weight = 1 - pos_weight

        for pred in predicted_mask:
            pred = pred[mask]
            pred[pred == 0] = -1

            loss += torch.sum(gt_mask * (-torch.log(self.sigmoid(pred))) + (1 - binary_gt_mask) * (-torch.log(1 - self.sigmoid(pred)))) / n

        return loss


class MaskLoss_CE_hed(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaskLoss_CE_hed, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_weight = loss_weight

    def forward(self, predicted_mask, gt_mask, depth_image):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        gt_mask = gt_mask[mask]

        binary_gt_mask = gt_mask.clone()
        binary_gt_mask[binary_gt_mask != 0] = 1.

        for pred in predicted_mask:
            pred = pred[mask]
            pred[pred == 0] = -1

            loss += torch.sum(gt_mask * (-torch.log(self.sigmoid(pred))) + (1 - binary_gt_mask) * (-torch.log(1 - self.sigmoid(pred)))) / n

        return loss


class BinaryMaskLoss(nn.Module):
    def __init__(self):
        super(BinaryMaskLoss, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, predicted_mask, gt_mask, depth_image, pos_weight=None):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        predicted_mask = predicted_mask[mask]
        gt_mask = gt_mask[mask]

        b, c, h, w = mask.size()
        pos_weight = 1.0 * n / (b * c * h * w)
        neg_weight = 1 - pos_weight

        if pos_weight != None:
            loss += torch.sum(pos_weight * gt_mask * (-torch.log(self.sigmoid(predicted_mask))) + neg_weight * (1 - gt_mask) * (-torch.log(1 - self.sigmoid(predicted_mask)))) / n
        else:
            loss += torch.sum(gt_mask * (-torch.log(self.sigmoid(predicted_mask))) + (1 - gt_mask) * (-torch.log(1 - self.sigmoid(predicted_mask)))) / n

        return loss


class MaskLoss_CE(nn.Module):
    def __init__(self):
        super(MaskLoss_CE, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, predicted_mask, gt_mask, depth_image, pos_weight=None, neg_weight=None):
        loss = 0
        mask = ~torch.isnan(depth_image)
        n = torch.sum(mask != False)
        predicted_mask = predicted_mask[mask]
        gt_mask = gt_mask[mask]

        binary_gt_mask = gt_mask.clone()
        binary_gt_mask[binary_gt_mask != 0] = 1.

        b, c, h, w = mask.size()
        pos_weight = 1.0 * n / (b * c * h * w)
        neg_weight = 1 - pos_weight

        if pos_weight != None and neg_weight != None:
            loss += torch.sum(pos_weight * gt_mask * (-torch.log(self.sigmoid(predicted_mask))) + neg_weight * (1 - binary_gt_mask) * (-torch.log(1 - self.sigmoid(predicted_mask)))) / n
        else:
            loss += torch.sum(gt_mask * (-torch.log(self.sigmoid(predicted_mask))) + (1 - binary_gt_mask) * (-torch.log(1 - self.sigmoid(predicted_mask)))) / n

        return loss


class L1LossWithImage(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1LossWithImage, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=NAMES['vgg19'])

    def forward(self, pred, target, weight=None, **kwargs):
        loss = 0
        for image in pred:
            feature = self.vgg_extractor(image.expand(3, -1, -1))
            loss += self.loss_weight * l1_loss(image, target, weight, reduction=self.reduction)
        
        return loss

                
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


def unnormalized_laplace_probability(value, location, diversity):
    return torch.exp(-torch.abs(location - value) / diversity) / (2 * diversity)


class SubpixelCrossEntropy(nn.Module):
    def __init__(self, diversity=1.0, disparity_step=2):
        super(SubpixelCrossEntropy, self).__init__()
        self._diversity = diversity
        self._disparity_step = disparity_step

    def forward(self, similarities, ground_truth_disparities, rank, weights=None):
        maximum_disparity_index = similarities.size(1)  # 32
        known_ground_truth_disparity = ground_truth_disparities.data != float('inf')
        # find the maximum similarity
        log_P_predicted = functional.log_softmax(similarities, dim=1)
        sum_P_target = torch.zeros(ground_truth_disparities.size()).to(rank)
        sum_P_target_x_log_P_predicted = torch.zeros(ground_truth_disparities.size()).to(rank)

        for disparity_index in range(maximum_disparity_index):
            disparity = disparity_index * self._disparity_step
            P_target = unnormalized_laplace_probability(
                value=disparity,
                location=ground_truth_disparities,
                diversity=self._diversity)
            sum_P_target += P_target
            sum_P_target_x_log_P_predicted += (log_P_predicted[:, disparity_index] * P_target)
        entropy = -sum_P_target_x_log_P_predicted[known_ground_truth_disparity] / sum_P_target[known_ground_truth_disparity]

        if weights is not None:
            weights_with_ground_truth = weights[known_ground_truth_disparity]
            return (weights_with_ground_truth * entropy).sum() / (weights_with_ground_truth.sum() + 1e-15)

        return entropy.mean()


def unnormalized_probability(gt, value, splits):
    gt_rounded = torch.round(gt)
    distri = torch.zeros_like(gt_rounded)
    mask = (gt_rounded != value) & (abs(gt_rounded - value) < splits)
    distri[mask] = (splits - abs(gt_rounded[mask] - value)) / splits
    distri[gt_rounded == value] = 1.
    return distri


class SubpixelCrossEntropyDiscrete(nn.Module):
    def __init__(self, disparity_step=2, network_splits=6):
        super(SubpixelCrossEntropyDiscrete, self).__init__()
        self.disparity_step = disparity_step
        self.network_splits = network_splits

    def forward(self, similarities, ground_truth_disparities):
        maximum_disparity_index = similarities.size(1)  # 32
        known_ground_truth_disparity = ground_truth_disparities.data != float('inf')

        sum_P_target = 0

        scaled_gt = ground_truth_disparities / self.disparity_step

        for disparity_index in range(maximum_disparity_index):
            P_target = unnormalized_probability(gt=scaled_gt, value=disparity_index, splits=self.network_splits)
            sum_P_target += torch.sum(torch.abs(similarities[:, disparity_index][known_ground_truth_disparity] - P_target[known_ground_truth_disparity]))

        return sum_P_target / (len(ground_truth_disparities[known_ground_truth_disparity]) * maximum_disparity_index)


class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, depth_img):
        loss = 0
        valid_mask = ~torch.isnan(depth_img)
        for p in pred:
            loss += self.loss_weight * l1_loss(p[valid_mask], target[valid_mask], reduction=self.reduction)
        return loss


class L2Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L2Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, depth_img):
        loss = 0
        valid_mask = ~torch.isnan(depth_img)
        for p in pred:
            loss += self.loss_weight * mse_loss(p[valid_mask], target[valid_mask], reduction=self.reduction)
        return loss


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction)


class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        y_diff = super(WeightedTVLoss, self).forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :], weight=weight[:, :, :-1, :])
        x_diff = super(WeightedTVLoss, self).forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:], weight=weight[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


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
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
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


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) -
                        self._gram_mat(gt_features[k]),
                        p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(
                        self._gram_mat(x_features[k]),
                        self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty
