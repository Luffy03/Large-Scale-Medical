import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import copy
import numpy as np


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, eval_bn=False):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        if eval_bn:
            self.bn1 = nn.BatchNorm3d(out_chan)
        else:
            self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        # print('bn', self.bn1.training, self.bn1.track_running_stats, self.bn1.affine)
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False, eval_bn=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act, eval_bn)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act, eval_bn)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act, eval_bn)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act, eval_bn)

    return nn.Sequential(layer1,layer2)


# class InputTransition(nn.Module):
#     def __init__(self, outChans, elu):
#         super(InputTransition, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = ContBatchNorm3d(16)
#         self.relu1 = ELUCons(elu, 16)
#
#     def forward(self, x):
#         # do we want a PRELU here as well?
#         out = self.bn1(self.conv1(x))
#         # split input in to 16 channels
#         x16 = torch.cat((x, x, x, x, x, x, x, x,
#                          x, x, x, x, x, x, x, x), 1)
#         out = self.relu1(torch.add(out, x16))
#         return out

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act, eval_bn=False):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act,eval_bn=eval_bn)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act, eval_bn=False):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True, eval_bn=eval_bn)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels, normalization):

        super(OutputTransition, self).__init__()

        # if up_sample:
        #     self.final_conv = nn.Sequential(
        #         nn.Conv3d(inChans, n_labels, kernel_size=1),
        #         nn.Upsample(scale_factor=(1, 2, 2), mode = 'trilinear'),
        #         nn.Sigmoid()
        #     )
        # else:
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)

        if normalization == 'sigmoid':
            assert n_labels == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert n_labels > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, x):
        out = self.normalization(self.final_conv(x))
        return out


class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, n_class=1, normalization='sigmoid', eval_bn=False, act='relu'):
        super(UNet3D, self).__init__()

        self.down_tr64 = DownTransition(in_channels,0,act, eval_bn)
        self.down_tr128 = DownTransition(64, 1, act, eval_bn)
        self.down_tr256 = DownTransition(128, 2, act, eval_bn)
        self.down_tr512 = DownTransition(256, 3, act, eval_bn)

        self.up_tr256 = UpTransition(512, 512, 2, act, eval_bn)
        self.up_tr128 = UpTransition(256, 256, 1, act, eval_bn)
        self.up_tr64 = UpTransition(128, 128, 0, act, eval_bn)
        self.out_tr = OutputTransition(64, n_class, normalization)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        return self.out

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        decoder_layers = ['up_tr256', 'up_tr128', 'up_tr64']
        out_layers = ['out_tr']
        module_dict = {'encoder': encoder_layers,
                       'decoder': decoder_layers,
                       'out': out_layers}
        return module_dict


class UNet3D_Encoder(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, act='relu', eval_bn=False):
        super(UNet3D_Encoder, self).__init__()

        self.down_tr64 = DownTransition(in_channels,0, act, eval_bn)
        self.down_tr128 = DownTransition(64, 1, act, eval_bn)
        self.down_tr256 = DownTransition(128, 2, act, eval_bn)
        self.down_tr512 = DownTransition(256, 3, act, eval_bn)


    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        return self.out512

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        module_dict = {'encoder': encoder_layers}
        return module_dict

class UNet3D_Dense(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, n_class=1, normalization='sigmoid', eval_bn=False, act='relu'):
        super(UNet3D_Dense, self).__init__()
        self.encoder = UNet3D_Encoder(in_channels=in_channels, act=act, eval_bn=eval_bn)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_class))

        if normalization == 'sigmoid':
            assert n_class == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert n_class > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, x):
        conv_x = self.encoder(x)
        # print(conv_x.size())
        dense_x = self.gap(conv_x)
        dense_x = torch.flatten(dense_x, 1, -1)
        logits = self.fc(dense_x)
        out = self.normalization(logits)
        return out

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        fc_layers = ['fc']
        module_dict = {'encoder': encoder_layers, 'fc:':fc_layers}
        return module_dict


# For RPL
class UNet3D_RPL(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, n_class=1, normalization='softmax', act='relu'):
        super(UNet3D_RPL, self).__init__()
        self.encoder = UNet3D_Encoder(in_channels=in_channels, act=act)
        self.gmp = torch.nn.AdaptiveMaxPool3d(1)
        self.fc6 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_class))

        if normalization == 'sigmoid':
            assert n_class == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert n_class > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward_once(self, x):
        conv_x = self.encoder(x)
        dense_x = self.gmp(conv_x)
        dense_x = torch.flatten(dense_x, 1, -1)
        logits = self.fc6(dense_x)
        return logits

    def forward(self, ref_patch, random_patch):
        output_fc6_ref = self.forward_once(ref_patch)
        output_fc6_random = self.forward_once(random_patch)
        output = torch.cat((output_fc6_ref, output_fc6_random), 1)
        output = self.fc(output)
        output = self.normalization(output)
        return output

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        fc_layers = ['fc']
        module_dict = {'encoder': encoder_layers, 'fc:':fc_layers}
        return module_dict


#Remove skip connection
class DownTransition_wo_skip(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition_wo_skip, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out

class UpTransition_wo_skip(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition_wo_skip, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(outChans,depth, act, double_chnnel=True)

    def forward(self, x):
        out_up_conv = self.up_conv(x)
        out = self.ops(out_up_conv)
        return out


class UNet3D_wo_skip(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, n_class=1, normalization='sigmoid', act='relu'):
        super(UNet3D_wo_skip, self).__init__()

        self.down_tr64 = DownTransition_wo_skip(in_channels,0,act)
        self.down_tr128 = DownTransition_wo_skip(64, 1, act)
        self.down_tr256 = DownTransition_wo_skip(128, 2, act)
        self.down_tr512 = DownTransition_wo_skip(256, 3, act)

        self.up_tr256 = UpTransition_wo_skip(512, 512, 2, act)
        self.up_tr128 = UpTransition_wo_skip(256, 256, 1, act)
        self.up_tr64 = UpTransition_wo_skip(128, 128, 0, act)
        self.out_tr = OutputTransition(64, n_class, normalization)

    def forward(self, x):
        self.out64 = self.down_tr64(x)
        self.out128 = self.down_tr128(self.out64)
        self.out256 = self.down_tr256(self.out128)
        self.out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512)
        self.out_up_128 = self.up_tr128(self.out_up_256)
        self.out_up_64 = self.up_tr64(self.out_up_128)
        self.out = self.out_tr(self.out_up_64)

        return self.out

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        decoder_layers = ['up_tr256', 'up_tr128', 'up_tr64']
        out_layers = ['out_tr']
        module_dict = {'encoder': encoder_layers,
                       'decoder': decoder_layers,
                       'out': out_layers}
        return module_dict


class UNet3D_JigSaw(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, order_n_class=100, num_cubes=8, act='relu'):
        super(UNet3D_JigSaw, self).__init__()
        self.encoder = UNet3D_Encoder(in_channels=in_channels, act=act)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.fc6 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.order_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, order_n_class)
        )

        self.num_cubes = num_cubes

    def forward_once(self, x):
        conv_x = self.encoder(x)
        dense_x = self.gap(conv_x)
        dense_x = torch.flatten(dense_x, 1, -1)
        logits = self.fc6(dense_x)
        return logits

    def forward(self, cubes):
        # [B, 8, C, X, Y, Z]
        cubes = cubes.transpose(0, 1)
        feats = []
        for i in range(self.num_cubes):
            output_fc6 = self.forward_once(cubes[i])
            feats.append(output_fc6)

        feats = torch.cat(feats, 1)
        # [B, K]
        order_logits = self.order_fc(feats)

        return order_logits

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        fc_layers = ['fc6', 'order_fc', 'hor_rot_fc', 'ver_rot_fc']
        module_dict = {'encoder': encoder_layers, 'fc':fc_layers}
        return module_dict


class UNet3D_RKB(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, order_n_class=100, num_cubes=8, act='relu'):
        super(UNet3D_RKB, self).__init__()
        self.encoder = UNet3D_Encoder(in_channels=in_channels, act=act)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.fc6 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.order_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, order_n_class)
        )

        self.ver_rot_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.hor_rot_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.num_cubes = num_cubes
        self.sigmoid = torch.nn.Sigmoid()

    def forward_once(self, x):
        conv_x = self.encoder(x)
        dense_x = self.gap(conv_x)
        dense_x = torch.flatten(dense_x, 1, -1)
        logits = self.fc6(dense_x)
        return logits

    def forward(self, cubes):
        # [B, 8, C, X, Y, Z]
        cubes = cubes.transpose(0, 1)
        # [8, B, C, X, Y, Z]
        feats = []
        for i in range(self.num_cubes):
            output_fc6 = self.forward_once(cubes[i])
            # hor_rot_logit = self.hor_rot_fc(output_fc6)
            # ver_rot_logit = self.ver_rot_fc(output_fc6)
            feats.append(output_fc6)
            ## hor_rot_logit: [B, 1]

        feats = torch.cat(feats, 1)
        # order_logits: [B, K]
        order_logits = self.order_fc(feats)
        # hor_rot_logits: [B*8, 1]
        hor_rot_logits = self.sigmoid(self.hor_rot_fc(feats))
        ver_rot_logits = self.sigmoid(self.ver_rot_fc(feats))

        return order_logits, hor_rot_logits, ver_rot_logits

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        fc_layers = ['fc6', 'order_fc', 'hor_rot_fc', 'ver_rot_fc']
        module_dict = {'encoder': encoder_layers, 'fc':fc_layers}
        return module_dict


class UNet3D_RKBP(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, order_n_class=100, num_cubes=8, act='relu'):
        super(UNet3D_RKBP, self).__init__()
        self.encoder = UNet3D_Encoder(in_channels=in_channels, act=act)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.fc6 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.order_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, order_n_class)
        )

        self.ver_rot_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.hor_rot_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.mask_fc = nn.Sequential(
            nn.Linear(num_cubes * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cubes))

        self.num_cubes = num_cubes
        self.sigmoid = torch.nn.Sigmoid()

    def forward_once(self, x):
        conv_x = self.encoder(x)
        dense_x = self.gap(conv_x)
        dense_x = torch.flatten(dense_x, 1, -1)
        logits = self.fc6(dense_x)
        return logits

    def forward(self, cubes):
        # [B, 8, C, X, Y, Z]
        cubes = cubes.transpose(0, 1)
        # [8, B, C, X, Y, Z]
        feats = []
        for i in range(self.num_cubes):
            output_fc6 = self.forward_once(cubes[i])
            # hor_rot_logit = self.hor_rot_fc(output_fc6)
            # ver_rot_logit = self.ver_rot_fc(output_fc6)
            feats.append(output_fc6)
            ## hor_rot_logit: [B, 1]

        feats = torch.cat(feats, 1)
        # order_logits: [B, K]
        order_logits = self.order_fc(feats)
        # hor_rot_logits: [B*8, 1]
        hor_rot_logits = self.sigmoid(self.hor_rot_fc(feats))
        ver_rot_logits = self.sigmoid(self.ver_rot_fc(feats))
        # mask
        mask_logits = self.sigmoid(self.mask_fc(feats))

        return order_logits, hor_rot_logits, ver_rot_logits, mask_logits

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        fc_layers = ['fc6', 'order_fc', 'hor_rot_fc', 'ver_rot_fc']
        module_dict = {'encoder': encoder_layers, 'fc':fc_layers}
        return module_dict


