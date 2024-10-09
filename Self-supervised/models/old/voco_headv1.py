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

    def forward(self, x):
        b, c = x.size()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x.view(b, self.out_dim)


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
        self.p = projection_head(in_dim=768)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feat(self, x_in):
        b = x_in.size()[0]
        hidden_states_out = self.swinViT(x_in)
        out = hidden_states_out[-1]

        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = self.p(out.view(b, -1))
        return out.view(b, -1)

    def forward(self, img, crops, labels):
        b = img.size()[0]
        inputs = torch.cat([img, crops], dim=0)
        outputs = self.forward_feat(inputs).as_tensor()
        feats, bases = outputs[:b], outputs[b:]

        logits = online_assign(feats, bases.detach())
        print('labels and logits:', labels[0].data, logits[0].data)

        loss = ce_loss(labels, logits)

        b_loss = regularization_loss(bases)
        print('b_loss:', b_loss.data)

        return loss + b_loss


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

    print('pos and neg:', pos_loss.data, neg_loss.data)

    loss = pos_loss + neg_loss
    return loss


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
    crops = torch.rand(16, 1, roi, roi, roi)
    # labels = torch.randint(low=0, high=16, size=(2, roi, roi, roi))
    labels = torch.rand(2, 16)
    model = VoCoHead(args)

    loss = model.forward(x, crops, labels)
    print(loss)
