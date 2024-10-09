# Copy from https://github.com/lswzjuer/NAS-WDAN/blob/68139048ef3fb2e9684cb9c60581367835c0fe9e/models/unet.py
import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F
import numpy as np


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet2D(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, normalization='sigmoid'):
        super(UNet2D, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        # nb_filter = [32,64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Up5 = up_conv(ch_in=nb_filter[4], ch_out=nb_filter[3])
        self.Up_conv5 = conv_block(ch_in=nb_filter[4], ch_out=nb_filter[3])

        self.Up4 = up_conv(ch_in=nb_filter[3], ch_out=nb_filter[2])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])

        self.Up3 = up_conv(ch_in=nb_filter[2], ch_out=nb_filter[1])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])

        self.Up2 = up_conv(ch_in=nb_filter[1], ch_out=nb_filter[0])
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1, stride=1, padding=0)

        if normalization == 'sigmoid':
            # assert output_ch == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert output_ch > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.normalization(d1)

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']
        decoder_layers = ['Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Up2', 'Up_conv2']
        out_layers = ['Conv_1x1']
        module_dict = {'encoder': encoder_layers,
                       'decoder': decoder_layers,
                       'out': out_layers}

        return module_dict


class UNet2D_Dense(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, normalization=None):
        super(UNet2D_Dense, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        hidden_size = 2048
        self.num_class = output_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(2048, self.num_class))

   
        if normalization == 'sigmoid':
            # assert self.num_class == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert self.num_class > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, x):
        # encoding path
        features = []
        x1 = self.Conv1(x)
        features.append(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        features.append(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        features.append(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        features.append(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        features.append(x5)

        # decoding + concat path
        x5_pool = self.Avgpool(x5)
        # x5_pool = self.maxpool(x5)
        x6 = torch.flatten(x5_pool, 1)
        # out = self.projector(x6)
        out = self.fc(x6)

        return out

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']
        fc_layers = ['fc']
        module_dict = {'encoder': encoder_layers,
                       'fc': fc_layers}
        return module_dict


class U_Net_Encoder(nn.Module):
    def __init__(self, img_ch=1, projection_size=128):
        super(U_Net_Encoder, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        hidden_size = nb_filter[4]
        projection_size = projection_size
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.projector = nn.Sequential(
        #     nn.Linear(1024, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_size, projection_size)
        # )

    def forward(self, x):
        # encoding path
        features = []
        x1 = self.Conv1(x)
        features.append(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        features.append(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        features.append(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        features.append(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        features.append(x5)

        # decoding + concat path
        x5_pool = self.Avgpool(x5)
        # representation = torch.flatten(x5_pool, 1)
        # representation = self.projector(representation)

        return x5_pool


class UNet2D_RPL(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, normalization=None):
        super( UNet2D_RPL, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        hidden_size = 2048
        self.num_class = output_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc6 = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size),
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_size, 8)
        )
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert self.num_class > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward_once(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # decoding + concat path
        x5_pool = self.Avgpool(x5)
        output = x5_pool.view(x5_pool.size()[0], -1)
        output = self.fc6(output)
        return output

    def forward(self, uniform_patch, random_patch):
        output_fc6_uniform = self.forward_once(uniform_patch)
        output_fc6_random = self.forward_once(random_patch)
        output = torch.cat((output_fc6_uniform, output_fc6_random), 1)
        output = self.fc(output)
        return output


class UNet2D_JigSaw(nn.Module):
    def __init__(self, im_ch=1, output_ch=100, num_cubes=9, normalization=None):
        super(UNet2D_JigSaw, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        hidden_size = 4096
        self.num_class = output_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=im_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc6 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.order_fc = nn.Sequential(
            nn.Linear(num_cubes * 512, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_class)
        )
        self.num_cubes = num_cubes

        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert self.num_class > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward_once(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        dense_x = self.Avgpool(x5)
        dense_x = torch.flatten(dense_x, 1, -1)
        logits = self.fc6(dense_x)
        return logits

    def forward(self, cubes):
        # [B, 9, C, X, Y]
        cubes = cubes.transpose(0, 1)
        # [9, B, C, X, Y]
        feats = []
        for i in range(self.num_cubes):
            output_fc6 = self.forward_once(cubes[i])
            feats.append(output_fc6)

        feats = torch.cat(feats, 1)
        # [B, K]
        order_logits = self.order_fc(feats)

        return order_logits
