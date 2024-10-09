# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op


class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Swin(nn.Module):
    def __init__(self, args):
        super(Swin, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=16 * args.feature_size,
        #     out_channels=8 * args.feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size * 8,
        #     out_channels=args.feature_size * 4,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder3 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size * 4,
        #     out_channels=args.feature_size * 2,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        # self.decoder2 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size * 2,
        #     out_channels=args.feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )
        #
        # self.decoder1 = UnetrUpBlock(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.feature_size,
        #     out_channels=args.feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in):
        b = x_in.size()[0]
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        encs = [enc0, enc1, enc2, enc3, dec4]

        # for enc in encs:
        #     print(enc.shape)

        out = self.forward_encs(encs)
        return out.view(b, -1)


class VoCoHead(nn.Module):
    def __init__(self, args):
        super(VoCoHead, self).__init__()
        self.backbone = Swin(args)

        hidden_dim, out_dim = 1024, 1024
        if args.feature_size == 48:
            in_dim = 1152
        elif args.feature_size == 96:
            in_dim = 2304
        else:
            in_dim = 4608
        self.student = projection_head(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.teacher = projection_head(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        ## no scheduler here
        momentum = 0.9
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def forward(self, img, crops, labels):
        batch_size = labels.size()[0]
        total_size = img.size()[0]
        sw_size = total_size // batch_size

        # loss accumulate
        intra, inter, total_b_loss = 0.0, 0.0, 0.0

        img, crops = img.as_tensor(), crops.as_tensor()
        inputs = torch.cat([img, crops], dim=0)

        # here we do norm on all instances
        embeddings = self.backbone(inputs)

        # feature augmentation
        aug_embeddings = nn.Dropout1d(0.2)(embeddings)
        student = self.student(aug_embeddings)

        self._EMA_update_encoder_teacher()
        with torch.no_grad():
            teacher = self.teacher(embeddings)

        x_student, bases_student = student[:total_size], student[total_size:]
        x_teacher, bases_teacher = teacher[:total_size], teacher[total_size:]

        for i in range(batch_size):
            label = labels[i]
            bases_num = 9

            x_stu, bases_stu = x_student[i * sw_size:(i + 1) * sw_size], bases_student[i * bases_num:(i + 1) * bases_num]
            x_tea, bases_tea = x_teacher[i * sw_size:(i + 1) * sw_size], bases_teacher[i * bases_num:(i + 1) * bases_num]
            logits = online_assign(x_stu, bases_tea)

            # if i == 0:
            #     print('labels and logits:', label[0].data, logits[0].data)

            intra_loss = ce_loss(label, logits)
            intra += intra_loss

            # teacher bases for inter volume contrast
            # j: different case
            j = (i + 1) % batch_size
            inter_bases_stu = bases_student[j * bases_num:(j + 1) * bases_num]
            inter_bases_tea = bases_teacher[j * bases_num:(j + 1) * bases_num]

            inter_loss = self.inter_volume(x_stu, x_tea, inter_bases_stu, inter_bases_tea)
            inter += inter_loss

            b_loss = regularization_loss(bases_stu)
            total_b_loss += b_loss

        intra = intra / batch_size
        inter = inter / batch_size
        total_b_loss = total_b_loss / batch_size

        loss = intra + inter + total_b_loss
        return loss

    def inter_volume(self, x_stu, x_tea, inter_bases_stu, inter_bases_tea):
        pred1 = online_assign(x_tea, inter_bases_tea)
        pred2 = online_assign(x_stu, inter_bases_stu)

        inter_loss = ce_loss(pred1.detach(), pred2)

        return inter_loss


def online_assign(feats, bases):
    b, c = feats.size()
    k, _ = bases.size()
    assert bases.size()[1] == c, print(feats.size(), bases.size())

    logits = []
    for i in range(b):
        feat = feats[i].unsqueeze(0)
        simi = F.cosine_similarity(feat, bases, dim=1).unsqueeze(0)
        logits.append(simi)
    logits = torch.concatenate(logits, dim=0)
    logits = F.relu(logits)

    return logits


def regularization_loss(bases):
    k, c = bases.size()
    loss_all = 0
    num = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            num += 1
            simi = F.cosine_similarity(bases[i].unsqueeze(0), bases[j].unsqueeze(0).detach(), dim=1)
            simi = F.relu(simi)
            loss_all += simi ** 2
    loss_all = loss_all / num

    return loss_all


def ce_loss(labels, logits):
    pos_dis = torch.abs(labels - logits)
    pos_loss = - labels * torch.log(1 - pos_dis + 1e-6)
    pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

    neg_lab = (labels == 0).long()
    neg_loss = - neg_lab * torch.log(1 - logits + 1e-6)
    neg_loss = neg_loss.sum() / (neg_lab.sum() + 1e-6)
    return pos_loss + neg_loss


if __name__ == '__main__':
    roi = 64
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=96, type=int, help="embedding size")
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

    args = parser.parse_args()
    base_num = 2
    bs_size = 2
    sw_size = 1

    x = torch.rand(bs_size*sw_size, 1, roi, roi, roi)
    crops = torch.rand(bs_size*base_num, 1, roi, roi, roi)

    # labels = torch.randint(low=0, high=9, size=(2, roi, roi, roi))
    labels = torch.rand([bs_size, sw_size, base_num])

    model = VoCoHead(args)
    loss = model.forward(x, crops, labels)
    print(loss)
    #
    # back = Swin(args)
    # out = back(x)
    # print(out.shape)

    x = torch.rand(1, 128)
    x = nn.Dropout1d(0.1)(x)
    print(x.shape)

