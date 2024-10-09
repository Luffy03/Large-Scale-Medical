import torch
import torch.nn as nn
from networks.unet import conv_block, up_conv


# Modified  PCRL2d model according to the paper.


class PCRLEncoder2d(nn.Module):
    def __init__(self, n_class=1, act='relu', norm='bn', in_channels=1, student=False, low_dim=128, Version='V1'):
        super(PCRLEncoder2d, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=in_channels, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])
        self.Conv5 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[4])
        self.student = student
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(nb_filter[-1], low_dim)
        self.relu = nn.ReLU(inplace=True)
        self.student = student
        self.fc2 = nn.Linear(low_dim, low_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mixup=False):
        b = x.shape[0]
        if mixup:
            # just mapping the mixed out 512 to dense layer
            self.out512 = x
            feature = self.out512.clone()
            feature = self.avg_pool(feature)
            feature = feature.view(b, -1)
            feature = self.fc1(feature)
            feature = self.relu(feature)
            feature = self.fc2(feature)
            return feature
        else:
            x1 = self.Conv1(x)

            x2 = self.Maxpool(x1)
            x2 = self.Conv2(x2)

            x3 = self.Maxpool(x2)
            x3 = self.Conv3(x3)

            x4 = self.Maxpool(x3)
            x4 = self.Conv4(x4)

            x5 = self.Maxpool(x4)
            self.out = self.Conv5(x5)

            feature = self.out.clone()
            feature = self.avg_pool(feature)
            feature = feature.view(b, -1)
            feature = self.fc1(feature)
            feature = self.relu(feature)
            feature = self.fc2(feature)

            return feature, [self.out, x4, x3, x2, x1]


class PCRLDecoder2d(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, n_class=1, act='relu', student=True, norm='bn', Version='V1'):
        super(PCRLDecoder2d, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        self.Up5 = up_conv(ch_in=nb_filter[4], ch_out=nb_filter[3])
        self.Up_conv5 = conv_block(ch_in=nb_filter[4], ch_out=nb_filter[3])

        self.Up4 = up_conv(ch_in=nb_filter[3], ch_out=nb_filter[2])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])

        self.Up3 = up_conv(ch_in=nb_filter[2], ch_out=nb_filter[1])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])

        self.Up2 = up_conv(ch_in=nb_filter[1], ch_out=nb_filter[0])
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0], n_class, kernel_size=1, stride=1, padding=0)

        self.nb_filter = nb_filter
        self.aug_fc1 = nn.Linear(6, 256)
        self.aug_fc2 = nn.Linear(256, self.nb_filter[-1])
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats, aug_tensor):
        b = aug_tensor.shape[0]
        # To or Tm or Th
        aug_tensor = self.aug_fc1(aug_tensor)
        aug_tensor = self.relu(aug_tensor)
        aug_tensor = self.aug_fc2(aug_tensor)
        aug_tensor = self.sigmoid(aug_tensor)
        aug_tensor = aug_tensor.view(b, self.nb_filter[-1], 1, 1)

        x5, x4, x3, x2, x1 = feats
        x5 = x5 * aug_tensor
        # normal decoder
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

        out = self.Conv_1x1(d2)

        return out


