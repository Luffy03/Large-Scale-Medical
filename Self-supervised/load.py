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

import argparse
import os
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed

import torch
import torch.nn as nn
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'


parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name",
    default="model_bestVal_big.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")


from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from torch import nn
import torch


def get_Plain_nnUNet(num_input_channels=1, num_classes=21, deep_supervision=False):
    UNet_base_num_features = 32
    unet_max_num_features = 320

    conv_kernel_sizes=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
    pool_op_kernel_sizes = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
    n_conv_per_stage_encoder = [2,2,2,2,2,2]
    n_conv_per_stage_decoder = [2,2,2,2,2]

    dim = len(conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    num_stages = len(conv_kernel_sizes)

    conv_or_blocks_per_stage = {
                'n_conv_per_stage': n_conv_per_stage_encoder,
                'n_conv_per_stage_decoder': n_conv_per_stage_decoder
            }
    kwargs = {
                'PlainConvUNet': {
                    'conv_bias': True,
                    'norm_op': get_matching_instancenorm(conv_op),
                    'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                    'dropout_op': None, 'dropout_op_kwargs': None,
                    'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
                },
            }

    model = PlainConvUNet(
                input_channels=num_input_channels,
                n_stages=num_stages,
                features_per_stage=[min(UNet_base_num_features * 2 ** i,
                                        unet_max_num_features) for i in range(num_stages)],
                conv_op=conv_op,
                kernel_sizes=conv_kernel_sizes,
                strides=pool_op_kernel_sizes,
                num_classes=num_classes,
                deep_supervision=deep_supervision,
                **conv_or_blocks_per_stage,
                **kwargs['PlainConvUNet']
            )
    return model


def main():
    args = parser.parse_args()
    main_worker(args=args)


def main_worker(args):
    # model = Sw(args)
    from models.voco_head import Swin
    model = Swin(args)
    # root = './runs/logs_swin_B/'
    root = '/home/linshan/pretrained/'
    path = root + 'VoCo_B.pt'
    save_path = root + 'VoCo_B_head.pt'

    model = get_Plain_nnUNet()
    path = './checkpoint_best.pth'
    save_path = './VoComni_nnunet.pt'


    try:
        model_dict = torch.load(path, map_location=torch.device('cpu'))

        if "state_dict" in model_dict.keys():
            state_dict = model_dict["state_dict"]

        elif "network_weights" in model_dict.keys():
            state_dict = model_dict["network_weights"]

        elif "net" in model_dict.keys():
            state_dict = model_dict["net"]
        elif "student" in model_dict.keys():
            state_dict = model_dict["student"]
        else:
            state_dict = model_dict

        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)

        if "backbone." in list(state_dict.keys())[0]:
            print("Tag 'backbone.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

        if "swin_vit" in list(state_dict.keys())[0]:
            print("Tag 'swin_vit' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

        current_model_dict = model.state_dict()

        for k in current_model_dict.keys():
            if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
                print(k)

        new_state_dict = {
            k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else
            current_model_dict[k]
            for k in current_model_dict.keys()}

        model.load_state_dict(new_state_dict, strict=True)

    except ValueError:
        raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
