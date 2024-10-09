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
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F


class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
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


class VoCoHead(nn.Module):
    def __init__(self, args):
        super(VoCoHead, self).__init__()
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
            spatial_dims=args.spatial_dims
        )
        self.proj_head = projection_head(in_dim=768)
        # self.proj_head_t = projection_head(in_dim=768)

        # self.reg_head = projection_head(in_dim=768, hidden_dim=1024, out_dim=1024)

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        ## no scheduler here
        momentum = 0.9
        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def forward_feat(self, x_in):
        b = x_in.size()[0]

        hidden_states_out = self.swinViT(x_in)
        out = hidden_states_out[-1]

        out = F.relu(out)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        return out.view(b, -1)

    def forward(self, img, crops, labels):
        batch_size = labels.size()[0]

        total_size = img.size()[0]
        inputs = torch.cat([img, crops], dim=0)
        feats = self.forward_feat(inputs).as_tensor()

        x, bases = feats[:total_size], feats[total_size:]

        pos, neg, total_b_loss = 0.0, 0.0, 0.0
        sw_size = total_size // batch_size

        for i in range(batch_size):
            # We want to conduct norm on per instance !!!
            label = labels[i]

            x_, bases_ = x[i * sw_size:(i + 1) * sw_size], bases[i * 16:(i + 1) * 16]
            x_bases_ = torch.cat([x_, bases_], dim=0)
            x_bases_ = self.proj_head(x_bases_)
            x_proj, bases_proj = x_bases_[:sw_size], x_bases_[sw_size:]

            if i == 0:
                im1, im2 = img[0].mean(), img[1].mean()
                cro = crops[0].mean()
                print(im1.item(), im2.item(), cro.item())

                im1, im2 = x_[0].mean(), x_[1].mean()
                cro = bases[0].mean()
                print(im1.item(), im2.item(), cro.item())

                im1, im2 = x_proj[0].mean(), x_proj[1].mean()
                cro = bases_proj[0].mean()
                print(im1.item(), im2.item(), cro.item())

            logits = online_assign(x_proj, bases_proj.detach())

            if i == 0:
                print('labels and logits:', label[0].data, logits[0].data)

            pos_loss, neg_loss = ce_loss(label, logits)
            pos += pos_loss
            neg += neg_loss

            b_loss = regularization_loss(bases_proj)
            total_b_loss += b_loss

        pos, neg = pos / batch_size, neg / batch_size
        total_b_loss = total_b_loss / batch_size

        return pos, neg, total_b_loss


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
    pos_lab = (labels > 0).long()
    pos_dis = torch.abs(labels - logits)
    pos_loss = - pos_lab * torch.log(1 - pos_dis + 1e-6)
    pos_loss = pos_loss.sum() / (pos_lab.sum() + 1e-6)

    neg_lab = (labels == 0).long()
    neg_loss = neg_lab * (logits ** 2)
    neg_loss = neg_loss.sum() / (neg_lab.sum() + 1e-6)
    return pos_loss, neg_loss


if __name__ == '__main__':
    roi = 48
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

    args = parser.parse_args()
    x = torch.rand(2, 1, roi, roi, roi)
    crops = torch.rand(2, 1, roi, roi, roi)

    # labels = torch.randint(low=0, high=16, size=(2, roi, roi, roi))
    labels = torch.rand(2, 2)
    model = VoCoHead(args)

    loss = model.forward(x, crops, labels)
    print(loss)
