import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, layer, surrogate

from models.blocks import SEWResBlock, NNConvUpsampling, MultiplyBy, SEWResBlockLIF


class NeuromorphicNet(nn.Module):
    def __init__(self, surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.surrogate_fct = surrogate_function
        self.detach_rst = detach_reset
        self.v_th = v_threshold
        self.v_rst = v_reset

        self.max_test_accuracy = float('inf')
        self.epoch = 0

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                m.v.detach_()
            elif isinstance(m, layer.Dropout):
                m.mask.detach_()

    def get_network_state(self):
        state = []
        for m in self.modules():
            if hasattr(m, 'reset'):
                state.append(m.v)
        return state

    def change_network_state(self, new_state):
        module_index = 0
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.v = new_state[module_index]
                module_index += 1

    def set_output_potentials(self, new_pots):
        module_index = 0
        for m in self.modules():
            if isinstance(m, neuron.IFNode):
                m.v = new_pots[module_index]
                module_index += 1

    def increment_epoch(self):
        self.epoch += 1

    def get_max_accuracy(self):
        return self.max_test_accuracy

    def update_max_accuracy(self, new_acc):
        self.max_test_accuracy = new_acc

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StereoSpike(NeuromorphicNet):
    """
    Baseline model, with which we report state-of-the-art performances in the second version of our paper.

    - all neuron potentials must be reset at each timestep
    - predict_depth layers do have biases, but it is equivalent to remove them and reset output I-neurons to the sum of all 4 biases, instead of 0.
    """
    def __init__(self, surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.0, multiply_factor=1.):
        super().__init__(surrogate_function=surrogate_function, detach_reset=detach_reset)

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor),
            SEWResBlock(512, v_threshold=v_threshold, v_reset=v_reset, connect_function='ADD', multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=0.0, surrogate_function=surrogate_function)

    def forward(self, x, save_spike_tensors=False):
        # x must be of shape [batch_size, num_frames_per_depth_map, 4 (2 cameras - 2 polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = self.bottom(frame)[0]
        # print(out_bottom.requires_grad)

        # pass through encoder layers
        out_conv1 = self.conv1(out_bottom)[0]
        out_conv2 = self.conv2(out_conv1)[0]
        out_conv3 = self.conv3(out_conv2)[0]
        out_conv4 = self.conv4(out_conv3)[0]

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv)[0]
        out_add4 = out_deconv4 + out_conv3
        self.Ineurons(self.predict_depth4(out_add4))
        depth4 = self.Ineurons.v

        out_deconv3 = self.deconv3(out_add4)[0]
        out_add3 = out_deconv3 + out_conv2
        self.Ineurons(self.predict_depth3(out_add3))
        depth3 = self.Ineurons.v

        out_deconv2 = self.deconv2(out_add3)[0]
        out_add2 = out_deconv2 + out_conv1
        self.Ineurons(self.predict_depth2(out_add2))
        depth2 = self.Ineurons.v

        out_deconv1 = self.deconv1(out_add2)[0]
        out_add1 = out_deconv1 + out_bottom
        self.Ineurons(self.predict_depth1(out_add1))
        depth1 = self.Ineurons.v

        # the membrane potentials of the output IF neuron carry the depth prediction
        # also output intermediate spike tensors
        return [depth1, depth2, depth3, depth4], [], [out_rconv, out_add4, out_add3, out_add2, out_add1], [], [out_conv1, out_conv2, out_conv3, out_conv4]

    def calculate_firing_rates(self, x):

        # dictionary to store the firing rates for all layers
        firing_rates_dict = {
            'encoder': 0.,
            'decoder': 0.,
            'bottleneck': 0.,
        }

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer and passes through encoder layers
        encoder_all = 0
        encoder_spikes = 0

        out_bottom = self.bottom(frame)[0]
        encoder_spikes += torch.sum(out_bottom)
        encoder_all += out_bottom.numel()

        out_conv1 = self.conv1(out_bottom)[0]
        encoder_spikes += torch.sum(out_conv1)
        encoder_all += out_conv1.numel()

        out_conv2 = self.conv2(out_conv1)[0]
        encoder_spikes += torch.sum(out_conv2)
        encoder_all += out_conv2.numel()

        out_conv3 = self.conv3(out_conv2)[0]
        encoder_spikes += torch.sum(out_conv3)
        encoder_all += out_conv3.numel()

        out_conv4 = self.conv4(out_conv3)[0]
        encoder_spikes += torch.sum(out_conv4)
        encoder_all += out_conv4.numel()

        firing_rates_dict['encoder'] = encoder_spikes / encoder_all

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)
        btn_spikes = torch.sum(out_rconv)
        btn_all = out_rconv.numel()
        firing_rates_dict['bottleneck'] = btn_spikes / btn_all

        # gradually upsample while concatenating and passing through skip connections
        decoder_all = 0
        decoder_spks = 0

        out_deconv4 = self.deconv4(out_rconv)[0]
        decoder_all += out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3
        decoder_spks += torch.sum(out_add4 != 0)

        out_deconv3 = self.deconv3(out_add4)[0]
        decoder_all += out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2
        decoder_spks += torch.sum(out_add3 != 0)

        out_deconv2 = self.deconv2(out_add3)[0]
        decoder_all += out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1
        decoder_spks += torch.sum(out_add2 != 0)

        out_deconv1 = self.deconv1(out_add2)[0]
        decoder_all += out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom
        decoder_spks += torch.sum(out_add1 != 0)

        firing_rates_dict['decoder'] = 1.0 * decoder_spks / decoder_all

        return firing_rates_dict

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior
    
    def calculate_computation(self, x):
        flops = 0

        frame = x[:, 0, :, :, :]
        out_bottom = self.bottom(frame)[0]
        flops += 1.0 * (4 * 5 * 5 - 1) * 260 * 346 * 32 * torch.sum(frame != 0) / frame.numel() + out_bottom.numel()

        out_conv1 = self.conv1(out_bottom)[0]
        flops += 1.0 * (32 * 5 * 5 - 1) * 130 * 173 * 64 * torch.sum(out_bottom) / out_bottom.numel() + out_conv1.numel()

        out_conv2 = self.conv2(out_conv1)[0]
        flops += 1.0 * (64 * 5 * 5 - 1) * 65 * 87 * 128 * torch.sum(out_conv1) / out_conv1.numel() + out_conv2.numel()

        out_conv3 = self.conv3(out_conv2)[0]
        flops += 1.0 * (128 * 5 * 5 - 1) * 33 * 44 * 256 * torch.sum(out_conv2) / out_conv2.numel() + out_conv3.numel()

        out_conv4 = self.conv4(out_conv3)[0]
        flops += 1.0 * (256 * 5 * 5 - 1) * 17 * 22 * 512 * torch.sum(out_conv3) / out_conv3.numel() + out_conv4.numel()

        out_rconv = self.bottleneck(out_conv4)
        flops += 1.0 * (512 * 3 * 3 - 1) * 17 * 22 * 512 * torch.sum(out_conv4) / out_conv4.numel() + out_rconv.numel()

        # -------------------------------- deconv4
        out_deconv4 = self.deconv4(out_rconv)[0]
        flops += 1.0 * (512 * 3 * 3 - 1) * 33 * 44 * 256 * torch.sum(out_rconv) / out_rconv.numel() + out_deconv4.numel()

        # adding operations
        flops += out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3
        
        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add4 != 0) / out_add4.numel()
        depth4_weight = self.predict_depth4(out_add4)
        self.Ineurons(depth4_weight)
        
        # -------------------------------- deconv3
        out_deconv3 = self.deconv3(out_add4)[0]
        flops += 1.0 * (256 * 3 * 3 - 1) * 65 * 87 * 128 * torch.sum(out_deconv4) / out_deconv4.numel() + out_deconv3.numel()

        flops += out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2

        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add3 != 0) / out_add3.numel()
        depth3_weight = self.predict_depth3(out_add3)
        self.Ineurons(depth3_weight)
        # accumulating potential
        flops += 260 * 346

        # -------------------------------- deconv2
        out_deconv2 = self.deconv2(out_add3)[0]
        flops += 1.0 * (128 * 3 * 3 - 1) * 130 * 173 * 64 * torch.sum(out_deconv3) / out_deconv3.numel() + out_deconv2.numel()
        
        flops += out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1
        
        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add2 != 0) / out_add2.numel()
        depth2_weight = self.predict_depth2(out_add2)
        self.Ineurons(depth2_weight)
        # accumulating potential
        flops += 260 * 346

        # -------------------------------- deconv1
        out_deconv1 = self.deconv1(out_add2)[0]
        flops += 1.0 * (64 * 3 * 3 - 1) * 260 * 346 * 32 * torch.sum(out_deconv2) / out_deconv2.numel() + out_deconv1.numel()

        flops += out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom

        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add1 != 0) / out_add1.numel()
        depth1_weight = self.predict_depth1(out_add1)
        self.Ineurons(depth1_weight)
        # accumulating potential
        flops += 260 * 346

        return flops


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


class StereoSpikeWithEdgeBinaryAndImage(NeuromorphicNet):
    """
    Baseline model, with which we report state-of-the-art performances in the second version of our paper.

    - all neuron potentials must be reset at each timestep
    - predict_depth layers do have biases, but it is equivalent to remove them and reset output I-neurons to the sum of all 4 biases, instead of 0.
    """
    def __init__(self, surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.0, multiply_factor=1.):
        super().__init__(surrogate_function=surrogate_function, detach_reset=detach_reset)

        # bottom layer, preprocessing the input spike frame without downsampling
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # encoder layers (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # residual layers
        self.bottleneck = nn.Sequential(
            SEWResBlock(512, v_threshold=self.v_th, v_reset=self.v_rst, connect_function='ADD', multiply_factor=multiply_factor),
            SEWResBlock(512, v_threshold=self.v_th, v_reset=self.v_rst, connect_function='ADD', multiply_factor=multiply_factor),
        )

        # decoder layers (upsampling)
        self.deconv4 = nn.Sequential(
            NNConvUpsampling(in_channels=512, out_channels=256, kernel_size=5, up_size=(33, 44)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        self.deconv3 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=128, kernel_size=5, up_size=(65, 87)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        self.deconv2 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=64, kernel_size=5, up_size=(130, 173)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )
        
        self.deconv1 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=32, kernel_size=5, up_size=(260, 346)),
            MultiplyBy(multiply_factor),
            neuron.IFNode(v_threshold=self.v_th, v_reset=self.v_rst, surrogate_function=self.surrogate_fct, detach_reset=True),
        )

        # these layers output depth maps at different scales, where depth is represented by the potential of IF neurons
        # that do not fire ("I-neurons"), i.e., with an infinite threshold.
        self.predict_depth4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_depth3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_depth2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_depth1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.Ineurons = neuron.IFNode(v_threshold=float('inf'), v_reset=0.0, surrogate_function=self.surrogate_fct, detach_reset=True)

        self.predict_mask4 = nn.Sequential(
            NNConvUpsampling(in_channels=256, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_mask3 = nn.Sequential(
            NNConvUpsampling(in_channels=128, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_mask2 = nn.Sequential(
            NNConvUpsampling(in_channels=64, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_mask1 = nn.Sequential(
            NNConvUpsampling(in_channels=32, out_channels=1, kernel_size=3, up_size=(260, 346), bias=True),
            MultiplyBy(multiply_factor),
        )

        self.predict_depth_mask_neuron = neuron.IFNode(v_threshold=self.v_th, v_reset=0.0, surrogate_function=self.surrogate_fct, detach_reset=True)

        self.feature_extractor1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU())

        self.feature_extractor2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=1, padding=0, bias=True),
            nn.ReLU())

        self.feature_extractor3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ReLU())

        self.feature_extractor4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ReLU())

    def forward(self, x):
        # x must be of shape [batch_size, num_frames_per_depth_map, 4 (2 cameras - 2 polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer
        out_bottom = self.bottom(frame)[0]

        # pass through encoder layers
        out_conv1 = self.conv1(out_bottom)[0]
        out_conv2 = self.conv2(out_conv1)[0]
        out_conv3 = self.conv3(out_conv2)[0]
        out_conv4 = self.conv4(out_conv3)[0]

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)

        # gradually upsample while concatenating and passing through skip connections
        out_deconv4 = self.deconv4(out_rconv)[0]
        out_add4 = out_deconv4 + out_conv3
        depth4_weight = self.predict_depth4(out_add4)
        self.Ineurons(depth4_weight)
        edge_mask4, v_4 = self.predict_depth_mask_neuron(self.predict_mask4(out_add4))
        depth4 = self.Ineurons.v + v_4

        out_deconv3 = self.deconv3(out_add4)[0]
        out_add3 = out_deconv3 + out_conv2
        depth3_weight = self.predict_depth3(out_add3)
        self.Ineurons(depth3_weight)
        edge_mask3, v_3 = self.predict_depth_mask_neuron(self.predict_mask3(out_add3))
        depth3 = self.Ineurons.v + v_3

        out_deconv2 = self.deconv2(out_add3)[0]
        out_add2 = out_deconv2 + out_conv1
        depth2_weight = self.predict_depth2(out_add2)
        self.Ineurons(depth2_weight)
        edge_mask2, v_2 = self.predict_depth_mask_neuron(self.predict_mask2(out_add2))
        depth2 = self.Ineurons.v + v_2

        out_deconv1 = self.deconv1(out_add2)[0]
        out_add1 = out_deconv1 + out_bottom
        depth1_weight = self.predict_depth1(out_add1)
        self.Ineurons(depth1_weight)
        edge_mask1, v_1 = self.predict_depth_mask_neuron(self.predict_mask1(out_add1))
        depth1 = self.Ineurons.v + v_1

        feature1 = self.feature_extractor1(out_conv1)
        feature2 = self.feature_extractor2(out_conv2)
        feature3 = self.feature_extractor3(out_conv3)
        feature4 = self.feature_extractor4(out_conv4)

        return [depth1, depth2, depth3, depth4], [edge_mask1, edge_mask2, edge_mask3, edge_mask4], [v_1, v_2, v_3, v_4], [feature1, feature2, feature3, feature4], []

    def calculate_firing_rates(self, x):
        # dictionary to store the firing rates for all layers
        firing_rates_dict = {
            'encoder': 0.,
            'decoder': 0.,
            'bottleneck': 0.,
            'EPM': 0.,
        }

        # x must be of shape [batch_size, num_frames_per_depth_map, 2 (polarities), W, H]
        frame = x[:, 0, :, :, :]

        # data is fed in through the bottom layer and passes through encoder layers
        encoder_all = 0
        encoder_spikes = 0

        out_bottom = self.bottom(frame)[0]
        encoder_spikes += torch.sum(out_bottom)
        encoder_all += out_bottom.numel()

        out_conv1 = self.conv1(out_bottom)[0]
        encoder_spikes += torch.sum(out_conv1)
        encoder_all += out_conv1.numel()

        out_conv2 = self.conv2(out_conv1)[0]
        encoder_spikes += torch.sum(out_conv2)
        encoder_all += out_conv2.numel()

        out_conv3 = self.conv3(out_conv2)[0]
        encoder_spikes += torch.sum(out_conv3)
        encoder_all += out_conv3.numel()

        out_conv4 = self.conv4(out_conv3)[0]
        encoder_spikes += torch.sum(out_conv4)
        encoder_all += out_conv4.numel()

        firing_rates_dict['encoder'] = encoder_spikes / encoder_all

        # pass through residual blocks
        out_rconv = self.bottleneck(out_conv4)
        btn_spikes = torch.sum(out_rconv)
        btn_all = out_rconv.numel()
        firing_rates_dict['bottleneck'] = btn_spikes / btn_all

        # gradually upsample while concatenating and passing through skip connections
        decoder_all = 0
        decoder_spks = 0

        out_deconv4 = self.deconv4(out_rconv)[0]
        decoder_all += out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3
        decoder_spks += torch.sum(out_add4 != 0)
        edge_mask4, v_4 = self.predict_depth_mask_neuron(self.predict_mask4(out_add4))

        out_deconv3 = self.deconv3(out_add4)[0]
        decoder_all += out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2
        decoder_spks += torch.sum(out_add3 != 0)
        edge_mask3, v_3 = self.predict_depth_mask_neuron(self.predict_mask3(out_add3))

        out_deconv2 = self.deconv2(out_add3)[0]
        decoder_all += out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1
        decoder_spks += torch.sum(out_add2 != 0)
        edge_mask2, v_2 = self.predict_depth_mask_neuron(self.predict_mask2(out_add2))

        out_deconv1 = self.deconv1(out_add2)[0]
        decoder_all += out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom
        decoder_spks += torch.sum(out_add1 != 0)
        edge_mask1, v_1 = self.predict_depth_mask_neuron(self.predict_mask1(out_add1))
        firing_rates_dict['decoder'] = 1.0 * decoder_spks / decoder_all

        firing_rates_dict['EPM'] = torch.sum(edge_mask1) / edge_mask1.numel()

        return firing_rates_dict

    def set_init_depths_potentials(self, depth_prior):
        self.Ineurons.v = depth_prior
    
    def calculate_computation(self, x):
        flops = 0

        frame = x[:, 0, :, :, :]
        out_bottom = self.bottom(frame)[0]
        flops += 1.0 * (4 * 5 * 5 - 1) * 260 * 346 * 32 * torch.sum(frame != 0) / frame.numel() + out_bottom.numel()

        out_conv1 = self.conv1(out_bottom)[0]
        flops += 1.0 * (32 * 5 * 5 - 1) * 130 * 173 * 64 * torch.sum(out_bottom) / out_bottom.numel() + out_conv1.numel()

        out_conv2 = self.conv2(out_conv1)[0]
        flops += 1.0 * (64 * 5 * 5 - 1) * 65 * 87 * 128 * torch.sum(out_conv1) / out_conv1.numel() + out_conv2.numel()

        out_conv3 = self.conv3(out_conv2)[0]
        flops += 1.0 * (128 * 5 * 5 - 1) * 33 * 44 * 256 * torch.sum(out_conv2) / out_conv2.numel() + out_conv3.numel()

        out_conv4 = self.conv4(out_conv3)[0]
        flops += 1.0 * (256 * 5 * 5 - 1) * 17 * 22 * 512 * torch.sum(out_conv3) / out_conv3.numel() + out_conv4.numel()

        out_rconv = self.bottleneck(out_conv4)
        flops += 1.0 * (512 * 3 * 3 - 1) * 17 * 22 * 512 * torch.sum(out_conv4) / out_conv4.numel() + out_rconv.numel()

        out_deconv4 = self.deconv4(out_rconv)[0]
        flops += 1.0 * (512 * 3 * 3 - 1) * 33 * 44 * 256 * torch.sum(out_rconv) / out_rconv.numel() + out_deconv4.numel()

        # adding operations
        flops += out_deconv4.numel()
        out_add4 = out_deconv4 + out_conv3
        
        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add4 != 0) / out_add4.numel()
        depth4_weight = self.predict_depth4(out_add4)
        self.Ineurons(depth4_weight)

        edge_mask4, v_4 = self.predict_depth_mask_neuron(self.predict_mask4(out_add4))
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add4 != 0) / out_add4.numel() + edge_mask4.numel()

        out_deconv3 = self.deconv3(out_add4)[0]
        flops += 1.0 * (256 * 3 * 3 - 1) * 65 * 87 * 128 * torch.sum(out_deconv4) / out_deconv4.numel() + out_deconv3.numel()

        flops += out_deconv3.numel()
        out_add3 = out_deconv3 + out_conv2

        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add3 != 0) / out_add3.numel()
        depth3_weight = self.predict_depth3(out_add3)
        self.Ineurons(depth3_weight)
        # accumulating potential
        flops += 260 * 346

        edge_mask3, v_3 = self.predict_depth_mask_neuron(self.predict_mask3(out_add3))
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add3 != 0) / out_add3.numel() + edge_mask3.numel() + edge_mask3.numel()

        out_deconv2 = self.deconv2(out_add3)[0]
        flops += 1.0 * (128 * 3 * 3 - 1) * 130 * 173 * 64 * torch.sum(out_deconv3) / out_deconv3.numel() + out_deconv2.numel()
        
        flops += out_deconv2.numel()
        out_add2 = out_deconv2 + out_conv1
        
        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add2 != 0) / out_add2.numel()
        depth2_weight = self.predict_depth2(out_add2)
        self.Ineurons(depth2_weight)
        # accumulating potential
        flops += 260 * 346

        edge_mask2, v_2 = self.predict_depth_mask_neuron(self.predict_mask2(out_add2))
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add2 != 0) / out_add2.numel() + edge_mask2.numel() + edge_mask2.numel()

        out_deconv1 = self.deconv1(out_add2)[0]
        flops += 1.0 * (64 * 3 * 3 - 1) * 260 * 346 * 32 * torch.sum(out_deconv2) / out_deconv2.numel() + out_deconv1.numel()

        flops += out_deconv1.numel()
        out_add1 = out_deconv1 + out_bottom

        # I-neuron
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add1 != 0) / out_add1.numel()
        depth1_weight = self.predict_depth1(out_add1)
        self.Ineurons(depth1_weight)
        # accumulating potential
        flops += 260 * 346

        edge_mask1, v_1 = self.predict_depth_mask_neuron(self.predict_mask1(out_add1))
        flops += 1.0 * (256 * 3 * 3) * 260 * 346 * torch.sum(out_add1 != 0) / out_add1.numel() + edge_mask1.numel() + edge_mask1.numel()

        # final add
        flops += 260 * 346
        return flops
