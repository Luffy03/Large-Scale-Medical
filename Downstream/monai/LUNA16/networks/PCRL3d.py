# Adapted from  https://github.com/Luchixiang/PCRL/
import torch
import torch.nn as nn


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, norm):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)

        if norm == 'bn':
            self.bn1 = nn.BatchNorm3d(num_features=out_chan, momentum=0.1, affine=True)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_chan, eps=1e-05, affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm3d(num_features=out_chan, momentum=0.1, affine=True)
        else:
            raise ValueError('normalization type {} is not supported'.format(norm))

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError('activation type {} is not supported'.format(act))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, norm, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act, norm)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act, norm)
    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), act, norm)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act, norm)

    return nn.Sequential(layer1, layer2)


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act, norm):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, norm, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        return self.ops(concat)


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act, norm):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act, norm)

    def forward(self, x):
        return self.ops(x)


class PCRLModel3d(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', norm='bn', in_channels=1, low_dim=128, student=False):
        super(PCRLModel3d, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.down_tr64 = DownTransition(in_channels, 0, act, norm)
        self.down_tr128 = DownTransition(64, 1, act, norm)
        self.down_tr256 = DownTransition(128, 2, act, norm)
        self.down_tr512 = DownTransition(256, 3, act, norm)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512, low_dim)
        self.relu = nn.ReLU(inplace=True)
        self.student = student
        self.fc2 = nn.Linear(low_dim, low_dim)
        self.up_tr256 = UpTransition(512, 512, 2, act, norm)
        self.up_tr128 = UpTransition(256, 256, 1, act, norm)
        self.up_tr64 = UpTransition(128, 128, 0, act, norm)
        self.out_tr = OutputTransition(64, n_class)
        self.aug_fc1 = nn.Linear(7, 256)
        self.aug_fc2 = nn.Linear(256, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feats_ema=None, alpha=None, aug_tensor1=None, aug_tensor2=None, mixup=False):
        b = x.shape[0]
        self.skip_out64 = self.down_tr64(x)
        self.skip_out128 = self.down_tr128(self.maxpool(self.skip_out64))
        self.skip_out256 = self.down_tr256(self.maxpool(self.skip_out128))
        self.out512 = self.down_tr512(self.maxpool(self.skip_out256))
        feature = self.out512.clone()
        feature = self.avg_pool(feature)
        feature = feature.view(b, -1)
        feature = self.fc1(feature)
        feature = self.relu(feature)
        feature = self.fc2(feature)
        if self.student:
            out512_ema, skip_out64_ema, skip_out128_ema, skip_out256_ema = feats_ema
            # alpha decoder
            if not mixup:
                aug_tensor1 = self.aug_fc1(aug_tensor1)
                aug_tensor1 = self.relu(aug_tensor1)
                aug_tensor1 = self.aug_fc2(aug_tensor1)
                aug_tensor2 = self.aug_fc1(aug_tensor2)
                aug_tensor2 = self.relu(aug_tensor2)
                aug_tensor2 = self.aug_fc2(aug_tensor2)
                aug_tensor1 = self.sigmoid(aug_tensor1)
                aug_tensor2 = self.sigmoid(aug_tensor2)
                aug_tensor1 = aug_tensor1.view(b, 512, 1, 1, 1)
                aug_tensor2 = aug_tensor2.view(b, 512, 1, 1, 1)
                self.out512 = self.out512 * aug_tensor1
                out512_ema = out512_ema * aug_tensor2
            out512 = self.out512 * alpha + (1 - alpha) * out512_ema
            skip_out256 = self.skip_out256 * alpha + (1 - alpha) * skip_out256_ema
            skip_out128 = self.skip_out128 * alpha + (1 - alpha) * skip_out128_ema
            skip_out64 = self.skip_out64 * alpha + (1 - alpha) * skip_out64_ema
            out_up_256_alpha = self.up_tr256(out512, skip_out256)
            out_up_128_alpha = self.up_tr128(out_up_256_alpha, skip_out128)
            out_up_64_alpha = self.up_tr64(out_up_128_alpha, skip_out64)
            out_alpha = self.out_tr(out_up_64_alpha)
            # normal decoder
            out_up_256 = self.up_tr256(self.out512, self.skip_out256)
            out_up_128 = self.up_tr128(out_up_256, self.skip_out128)
            out_up_64 = self.up_tr64(out_up_128, self.skip_out64)
            out = self.out_tr(out_up_64)
            return feature, out_alpha, out
        return feature, [self.out512, self.skip_out64, self.skip_out128, self.skip_out256]

    def decoder_shuffle(self, x, shuffle_num):
        w = x.shape[2]
        h = x.shape[3]
        d = x.shape[4]
        shuffle_col_index = torch.randperm(w)[:shuffle_num].cuda()
        shuffle_row_index = torch.randperm(h)[:shuffle_num].cuda()
        shuffle_d_index = torch.randperm(d)[:shuffle_num].cuda()
        col_index = shuffle_col_index[torch.randperm(shuffle_col_index.shape[0])]
        row_index = shuffle_row_index[torch.randperm(shuffle_row_index.shape[0])]
        d_index = shuffle_d_index[torch.randperm(shuffle_d_index.shape[0])]
        # print(col_index, row_index, shuffle_row_index, shuffle_col_index)
        # print(shuffle_row_index, x.shape, x[:, :, shuffle_row_index].shape)
        x = x.index_copy(2, col_index, x.index_select(2, shuffle_col_index))
        x = x.index_copy(3, row_index, x.index_select(3, shuffle_row_index))
        x = x.index_copy(4, d_index, x.index_select(4, shuffle_d_index))
        return x



