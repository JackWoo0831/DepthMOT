from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from dcn_v2 import DCN

from .pose_dla_dcn import BasicBlock, DLA, DLAUp, IDAUp

from lib.utils.depth_utils import transformation_from_parameters

"""
Depth part
"""

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
    

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        # self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            setattr(self, f'upconv_{i}_0', ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            setattr(self, f'upconv_{i}_1', ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            setattr(self, f'dispconv_{s}', Conv3x3(self.num_ch_dec[s], self.num_output_channels))

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = getattr(self, f'upconv_{i}_0')(x)
            x = [self._upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)  # channel dim * 2
            x = getattr(self, f'upconv_{i}_1')(x)
            if i in self.scales:
                self.outputs[f'disp_{i}'] = self.sigmoid(getattr(self, f'dispconv_{i}')(x))

        return self.outputs
    
    def _upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")
 
"""
Pose part
"""

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:  # conv1 concat several images, so rewrite the load weight
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225  # TODO: check if necessary
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
    
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        setattr(self, 'squeeze', nn.Conv2d(self.num_ch_enc[-1], 256, 1))

        setattr(self, 'pose_0', nn.Conv2d(num_input_features * 256, 256, 3, stride, 1))

        setattr(self, 'pose_1', nn.Conv2d(256, 256, 3, stride, 1))

        setattr(self, 'pose_2', nn.Conv2d(256, 6 * num_frames_to_predict_for, 1))

        self.relu = nn.ReLU()

        # self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):

        out = self.relu(self.squeeze(input_features[-1]))
        for i in range(3):
            out = getattr(self, f'pose_{i}')(out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


"""
final model
"""

class DLADepth(nn.Module):
    """
    DLA with depth branch
    """
    def __init__(self, heads, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, disp_scales=4, 
                 separate_pose=False, fit_mono_input=True):
        super(DLADepth, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))  # 2
        self.last_level = last_level  # 5
        self.base = DLA(levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512], block=BasicBlock)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

        # depth estimation
        # use first 4th output of DLA (self.base)
        self.disp_scales = disp_scales
        self.fit_mono_input = fit_mono_input 
        num_ch_enc = np.array([64, 64, 128, 256, 512]) if self.fit_mono_input else np.array([32, 64, 128, 256, 512])
        self.depth_decoder = DepthDecoder(num_ch_enc, scales=range(self.disp_scales),)

        # pose estimation
        if not separate_pose:
            self.pose_encoder = ResnetEncoder(num_layers=18, pretrained=True, num_input_images=2)
            self.pose_decoder = PoseDecoder(num_ch_enc=np.array([64, 64, 128, 256, 512]), num_input_features=1, 
                                            num_frames_to_predict_for=2)
        else:
            raise NotImplementedError


    def forward(self, x, x_prev, x_next):
        """
        Args:
            x: current frame, torch.Tensor (bs, 3, h, w)
            x_prev, x_next: previous and next frame for pose net

        Return:
            dict, key: item (center, offset, etc.), value: torch.Tensor
        
        """

        """
        After DLA: 
        torch.Size([1, 16, 608, 1088])
        torch.Size([1, 32, 304, 544])
        torch.Size([1, 64, 152, 272])
        torch.Size([1, 128, 76, 136])
        torch.Size([1, 256, 38, 68])
        torch.Size([1, 512, 19, 34])
        After DLA up: 
        torch.Size([1, 64, 152, 272])
        torch.Size([1, 128, 76, 136])
        torch.Size([1, 256, 38, 68])
        torch.Size([1, 512, 19, 34])
        After IDA up: 
        torch.Size([1, 64, 152, 272])
        torch.Size([1, 64, 152, 272])
        torch.Size([1, 64, 152, 272])
        """

        x_dla = self.base(x)

        # if fit mono input, concat twice torch.Size([1, 32, 304, 544])
        if self.fit_mono_input:            
            x_dla[1] = torch.cat([x_dla[1], x_dla[1]], dim=1)

        # depth estimation
        disp_output = self.depth_decoder(x_dla[-(self.disp_scales + 1): ])

        x_dlaup = self.dla_up(x_dla)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_dlaup[i].clone())
        self.ida_up(y, 0, len(y))

        output = {}
        for head in self.heads:
            output[head] = self.__getattr__(head)(y[-1])

        output.update(disp_output)

        # pose estimation   

        # cal transform matrix
        # TODO: why?
        pose_output = {}
        for i, frame_idx in enumerate([-1, 1]):
            
            if frame_idx < 0:
                pose_input = torch.cat([x_prev, x], dim=1)
            else:
                pose_input = torch.cat([x, x_next], dim=1)

            pose_encode = self.pose_encoder(pose_input)
            axisangle, translation = self.pose_decoder(pose_encode)

            pose_output[f'axis_0_{frame_idx}'] = axisangle
            pose_output[f'trans_0_{frame_idx}'] = translation
            pose_output[f'map_0_{frame_idx}'] = transformation_from_parameters(
                axisangle=axisangle[:, 0], translation=translation[:, 0], invert=(frame_idx < 0),
            )


        output.update(pose_output)
        return output
    
    def inference(self, x, x_prev):
        """
        in inference stage
        Args:
            x: current frame, torch.Tensor (bs, 3, h, w)
            x_prev: previous frame for pose net

        Return:
            dict, key: item (center, offset, etc.), value: torch.Tensor
        """
        x_dla = self.base(x)

        # if fit mono input, concat twice torch.Size([1, 32, 304, 544])
        if self.fit_mono_input:            
            x_dla[1] = torch.cat([x_dla[1], x_dla[1]], dim=1)

        # depth estimation
        disp_output = self.depth_decoder(x_dla[-(self.disp_scales + 1): ])

        x_dlaup = self.dla_up(x_dla)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_dlaup[i].clone())
        self.ida_up(y, 0, len(y))

        output = {}
        for head in self.heads:
            output[head] = self.__getattr__(head)(y[-1])

        output.update(disp_output)

        # pose estimation   

        # cal transform matrix from prev to now
        pose_output = {}

        if x_prev is not None:

            pose_input = torch.cat([x_prev, x], dim=1)
            pose_encode = self.pose_encoder(pose_input)
            axisangle, translation = self.pose_decoder(pose_encode)

            pose_output[f'axis_-1_0'] = axisangle
            pose_output[f'trans_-1_0'] = translation
            # when inferencing t-th frame, we cal the transformation from t-1 -> t, so invert=False
            pose_output[f'map_-1_0'] = transformation_from_parameters(
                axisangle=axisangle[:, 0], translation=translation[:, 0], invert=False,
                )
        else:
            pose_output[f'map_-1_0'] = None

        output.update(pose_output)
        return output

def get_dla_depth_net(num_layers, heads, head_conv=256, down_ratio=4):
  model = DLADepth(heads,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model