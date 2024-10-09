import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import torch
from segmentation_models_pytorch.base import modules as md
import numpy as np
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders._base import EncoderMixin
import copy
import random


def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class ShuffleUnetDecoder(nn.Module):
    def __init__(
            self,
            # decoder,
            encoder_channels=512,
            n_class=3,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None

    ):
        super().__init__()
        # self.decoder = decoder
        # self.segmentation_head = segmentation_head
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        # self.conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1)
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        initialize_decoder(self.blocks)
        # self.segmentation_head = SegmentationHead(16, 3)
        # initialize_head(self.segmentation_head)
        # self.segmentation_head = segmentation_head
        #
        # # combine decoder keyword arguments

    def forward(self, features1, features2, alpha, aug_tensor1, aug_tensor2, mixup=False):
        # x = self.decoder(*features)
        # return self.segmentation_head(x)
        # def forward(self, features1, features2):
        #
        features1 = features1[1:]  # remove first skip with same spatial resolution
        features1 = features1[::-1]  # reverse channels to start from head of encoder
        features2 = features2[1:]
        features2 = features2[::-1]
        head1 = features1[0]
        skips1 = features1[1:]
        head2 = features2[0]
        skips2 = features2[1:]
        x1 = self.center(head1)
        x2 = self.center(head2)
        if not mixup:
            x1 = x1 * aug_tensor1
            x2 = x2 * aug_tensor2
        x3 = x1.clone()
        x1 = alpha * x1 + (1 - alpha) * x2
        for i, decoder_block in enumerate(self.blocks):
            # print(i, x1.shape, skips1[i].shape, x2.shape, skips2[i].shape)
            skip1 = skips1[i] if i < len(skips1) else None
            #skip1_shuffle = self.decoder_shuffle(skip1, shuffle_num + i + 1) if i < len(skips1) else None
            x3 = decoder_block(x3, skip1)

            # x1 = decoder_block(x1, skip1)
            skip2 = skips2[i] if i < len(skips2) else None
            skip = alpha * skip1 + (1 - alpha) * skip2 if i < len(skips2) else None
            # skip = self.decoder_shuffle(skip, shuffle_num + i + 1) if i < len(skips2) else None
            # x2 = decoder_block(x2, skip2)
            x1 = decoder_block(x1, skip)

        # x1 = self.segmentation_head(x1)
        return x1, x3

    def decoder_shuffle(self, x, shuffle_num):
        w = x.shape[2]
        h = x.shape[3]
        shuffle_col_index = torch.randperm(w)[:shuffle_num].cuda()
        shuffle_row_index = torch.randperm(h)[:shuffle_num].cuda()
        col_index = shuffle_col_index[torch.randperm(shuffle_col_index.shape[0])]
        row_index = shuffle_row_index[torch.randperm(shuffle_row_index.shape[0])]
        # print(col_index, row_index, shuffle_row_index, shuffle_col_index)
        # print(shuffle_row_index, x.shape, x[:, :, shuffle_row_index].shape)
        x = x.index_copy(2, col_index, x.index_select(2, shuffle_col_index))
        x = x.index_copy(3, row_index, x.index_select(3, shuffle_row_index))
        return x


class PCRLModel(nn.Module):
    def __init__(self, n_class=3, low_dim=128, student=False):
        super(PCRLModel, self).__init__()
        self.model = smp.Unet('resnet18', in_channels=3, classes=n_class, encoder_weights=None)
        self.model.decoder = ShuffleUnetDecoder(self.model.encoder.out_channels)
        # self.segmentation_head = self.unet.segmentation_head
        # self.model = net
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, low_dim)
        self.relu = nn.ReLU(inplace=True)
        self.student = student
        self.fc2 = nn.Linear(low_dim, low_dim)
        self.aug_fc1 = nn.Linear(6, 256)
        self.aug_fc2 = nn.Linear(256, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, features_ema=None, alpha=None, aug_tensor1=None, aug_tensor2=None, mixup=False):
        b = x.shape[0]
        features = self.model.encoder(x)
        feature = self.avg_pool(features[-1])
        feature = feature.view(b, -1)
        feature = self.fc1(feature)
        feature = self.relu(feature)
        feature = self.fc2(feature)
        if self.student:
            if not mixup:
                aug_tensor1 = self.aug_fc1(aug_tensor1)
                aug_tensor1 = self.relu(aug_tensor1)
                aug_tensor1 = self.aug_fc2(aug_tensor1)
                aug_tensor2 = self.aug_fc1(aug_tensor2)
                aug_tensor2 = self.relu(aug_tensor2)
                aug_tensor2 = self.aug_fc2(aug_tensor2)
                aug_tensor1 = self.sigmoid(aug_tensor1)
                aug_tensor2 = self.sigmoid(aug_tensor2)
                aug_tensor1 = aug_tensor1.view(b, 512, 1, 1)
                aug_tensor2 = aug_tensor2.view(b, 512, 1, 1)
                # print(aug_tensor2.shape)
            decoder_output_alpha, decoder_output = self.model.decoder(features, features_ema, alpha, aug_tensor1,
                                                                      aug_tensor2, mixup)
            masks_alpha = self.model.segmentation_head(decoder_output_alpha)
            masks = self.model.segmentation_head(decoder_output)
            return feature, masks_alpha, masks
        return feature, features